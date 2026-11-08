//! Elementwise unary and broadcasting binary ops.

use super::grads;
use crate::backend::{BinaryOp, UnaryOp};
use crate::{DType, Error, Result, Shape, Tensor};

macro_rules! unary_fns {
    ($($(#[$m:meta])* $name:ident => $op:expr;)*) => {
        impl Tensor {
            $(
                $(#[$m])*
                pub fn $name(&self) -> Result<Tensor> {
                    self.unary($op)
                }
            )*
        }
    };
}

unary_fns! {
    /// `-x`.
    neg => UnaryOp::Neg;
    /// `|x|`.
    abs => UnaryOp::Abs;
    /// `e^x`.
    exp => UnaryOp::Exp;
    /// Natural logarithm.
    log => UnaryOp::Log;
    /// Square root.
    sqrt => UnaryOp::Sqrt;
    /// `1 / sqrt(x)`.
    rsqrt => UnaryOp::Rsqrt;
    /// `1 / x`.
    recip => UnaryOp::Recip;
    /// `x * x`.
    square => UnaryOp::Square;
    /// Sine.
    sin => UnaryOp::Sin;
    /// Cosine.
    cos => UnaryOp::Cos;
    /// Hyperbolic tangent.
    tanh => UnaryOp::Tanh;
    /// Arctangent.
    atan => UnaryOp::Atan;
    /// Arcsine (NaN outside `[-1, 1]`; the gradient is infinite at `+-1`).
    asin => UnaryOp::Asin;
    /// Arccosine (NaN outside `[-1, 1]`; the gradient is infinite at `+-1`).
    acos => UnaryOp::Acos;
    /// The error function.
    erf => UnaryOp::Erf;
    /// `1 / (1 + e^-x)`.
    sigmoid => UnaryOp::Sigmoid;
    /// `x * sigmoid(x)` (a.k.a. swish).
    silu => UnaryOp::Silu;
    /// `max(x, 0)`.
    relu => UnaryOp::Relu;
    /// `x * relu6(x + 3) / 6`.
    hardswish => UnaryOp::Hardswish;
    /// `ln(1 + e^x)`, stable for large `|x|`.
    softplus => UnaryOp::Softplus;
    /// Round down (zero gradient).
    floor => UnaryOp::Floor;
    /// Round up (zero gradient).
    ceil => UnaryOp::Ceil;
    /// Round half away from zero (zero gradient).
    round => UnaryOp::Round;
    /// -1, 0 or 1 (zero gradient).
    sign => UnaryOp::Sign;
}

impl Tensor {
    /// Apply an elementwise op (`F32`).
    pub fn unary(&self, op: UnaryOp) -> Result<Tensor> {
        self.check_f32(op.name())?;
        let b = self.backend()?;
        let s = self.with_view(|v| b.unary(op, v))?;
        let out = Tensor::wrap(s, self.shape().clone(), op.name())?;
        let x = self.clone();
        Ok(out.track(op.name(), &[self], move |g| Ok(grads![Some(x.unary_grad(op, g)?)])))
    }

    /// `g * op'(self)`, without autograd.
    pub(crate) fn unary_grad(&self, op: UnaryOp, g: &Tensor) -> Result<Tensor> {
        let b = self.backend()?;
        let s =
            super::with_views([self, g], [self.layout(), g.layout()], |[x, g]| b.unary_backward(op, x, g))?;
        Tensor::wrap(s, self.shape().clone(), op.name())
    }

    /// `x` for `x > 0`, else `slope * x`.
    pub fn leaky_relu(&self, slope: f32) -> Result<Tensor> {
        self.unary(UnaryOp::LeakyRelu(slope))
    }

    /// `min(max(x, lo), hi)`.
    pub fn clamp(&self, lo: f32, hi: f32) -> Result<Tensor> {
        if lo > hi {
            return Err(Error::invalid_arg("clamp", format!("min {lo} > max {hi}")));
        }
        self.unary(UnaryOp::Clamp(lo, hi))
    }

    /// `x^p` for a scalar exponent.
    pub fn powf(&self, p: f32) -> Result<Tensor> {
        self.unary(UnaryOp::PowScalar(p))
    }

    /// `x + c`.
    pub fn add_scalar(&self, c: f32) -> Result<Tensor> {
        self.unary(UnaryOp::AddScalar(c))
    }

    /// `x - c`.
    pub fn sub_scalar(&self, c: f32) -> Result<Tensor> {
        self.unary(UnaryOp::AddScalar(-c))
    }

    /// `x * c`.
    pub fn mul_scalar(&self, c: f32) -> Result<Tensor> {
        self.unary(UnaryOp::MulScalar(c))
    }

    /// `x / c`.
    pub fn div_scalar(&self, c: f32) -> Result<Tensor> {
        self.unary(UnaryOp::DivScalar(c))
    }

    /// `c - x`.
    pub fn rsub_scalar(&self, c: f32) -> Result<Tensor> {
        self.unary(UnaryOp::RSubScalar(c))
    }

    /// `c / x`.
    pub fn rdiv_scalar(&self, c: f32) -> Result<Tensor> {
        self.unary(UnaryOp::RDivScalar(c))
    }

    /// `max(x, c)`.
    pub fn maximum_scalar(&self, c: f32) -> Result<Tensor> {
        self.unary(UnaryOp::MaxScalar(c))
    }

    /// `min(x, c)`.
    pub fn minimum_scalar(&self, c: f32) -> Result<Tensor> {
        self.unary(UnaryOp::MinScalar(c))
    }

    /// Apply a broadcasting binary op (`F32`). Comparisons return 0/1 `F32` masks without gradient.
    pub fn binary(&self, op: BinaryOp, other: &Tensor) -> Result<Tensor> {
        let name = op.name();
        self.check_same_device(other, name)?;
        self.check_f32(name)?;
        other.check_f32(name)?;
        let b = self.backend()?;
        let (shape, s) =
            self.with_broadcast(other, name, |shape, va, vb| Ok((shape.clone(), b.binary(op, va, vb)?)))?;
        let out = Tensor::wrap(s, shape, name)?;
        if op.is_comparison() {
            return Ok(out);
        }
        let (x, y) = (self.clone(), other.clone());
        Ok(out.track(name, &[self, other], move |g| binary_backward(op, &x, &y, g)))
    }

    /// Sum `self` (a broadcast result) down to `shape`: the reverse of broadcasting.
    pub fn sum_to(&self, shape: &Shape) -> Result<Tensor> {
        if self.shape() == shape {
            return Ok(self.clone());
        }
        if !shape.broadcasts_to(self.shape()) {
            return Err(Error::ShapeMismatch { op: "sum_to", lhs: self.shape().clone(), rhs: shape.clone() });
        }
        let off = self.rank() - shape.rank();
        let dims: Vec<usize> =
            (0..self.rank()).filter(|&d| d < off || (shape[d - off] == 1 && self.dims()[d] != 1)).collect();
        let s = if dims.is_empty() { self.clone() } else { self.sum(dims, true)? };
        s.reshape(shape)
    }

    /// `self + other` with broadcasting.
    pub fn add(&self, other: &Tensor) -> Result<Tensor> {
        self.binary(BinaryOp::Add, other)
    }

    /// `self - other` with broadcasting.
    pub fn sub(&self, other: &Tensor) -> Result<Tensor> {
        self.binary(BinaryOp::Sub, other)
    }

    /// `self * other` with broadcasting.
    pub fn mul(&self, other: &Tensor) -> Result<Tensor> {
        self.binary(BinaryOp::Mul, other)
    }

    /// `self / other` with broadcasting.
    pub fn div(&self, other: &Tensor) -> Result<Tensor> {
        self.binary(BinaryOp::Div, other)
    }

    /// `self ^ other` with broadcasting.
    pub fn pow(&self, other: &Tensor) -> Result<Tensor> {
        self.binary(BinaryOp::Pow, other)
    }

    /// Elementwise maximum (ties split the gradient evenly).
    pub fn maximum(&self, other: &Tensor) -> Result<Tensor> {
        self.binary(BinaryOp::Maximum, other)
    }

    /// Elementwise minimum (ties split the gradient evenly).
    pub fn minimum(&self, other: &Tensor) -> Result<Tensor> {
        self.binary(BinaryOp::Minimum, other)
    }

    /// `atan2(self, other)` with broadcasting: the angle of the point `(x = other, y = self)` in `[-pi, pi]`.
    /// The gradient at the origin is taken as 0.
    pub fn atan2(&self, other: &Tensor) -> Result<Tensor> {
        self.binary(BinaryOp::Atan2, other)
    }

    /// `self == other` as a 0/1 `F32` mask.
    pub fn eq(&self, other: &Tensor) -> Result<Tensor> {
        self.binary(BinaryOp::Eq, other)
    }

    /// `self != other` as a 0/1 `F32` mask.
    pub fn ne(&self, other: &Tensor) -> Result<Tensor> {
        self.binary(BinaryOp::Ne, other)
    }

    /// `self > other` as a 0/1 `F32` mask.
    pub fn gt(&self, other: &Tensor) -> Result<Tensor> {
        self.binary(BinaryOp::Gt, other)
    }

    /// `self >= other` as a 0/1 `F32` mask.
    pub fn ge(&self, other: &Tensor) -> Result<Tensor> {
        self.binary(BinaryOp::Ge, other)
    }

    /// `self < other` as a 0/1 `F32` mask.
    pub fn lt(&self, other: &Tensor) -> Result<Tensor> {
        self.binary(BinaryOp::Lt, other)
    }

    /// `self <= other` as a 0/1 `F32` mask.
    pub fn le(&self, other: &Tensor) -> Result<Tensor> {
        self.binary(BinaryOp::Le, other)
    }

    /// A rank-0 `F32` tensor holding `v` on this tensor's device.
    pub(crate) fn scalar_like(&self, v: f32) -> Result<Tensor> {
        Tensor::full_with(Shape::scalar(), v as f64, DType::F32, self.device())
    }

    /// `self == c` as a 0/1 `F32` mask.
    pub fn eq_scalar(&self, c: f32) -> Result<Tensor> {
        self.eq(&self.scalar_like(c)?)
    }

    /// `self > c` as a 0/1 `F32` mask.
    pub fn gt_scalar(&self, c: f32) -> Result<Tensor> {
        self.gt(&self.scalar_like(c)?)
    }

    /// `self >= c` as a 0/1 `F32` mask.
    pub fn ge_scalar(&self, c: f32) -> Result<Tensor> {
        self.ge(&self.scalar_like(c)?)
    }

    /// `self < c` as a 0/1 `F32` mask.
    pub fn lt_scalar(&self, c: f32) -> Result<Tensor> {
        self.lt(&self.scalar_like(c)?)
    }

    /// `self <= c` as a 0/1 `F32` mask.
    pub fn le_scalar(&self, c: f32) -> Result<Tensor> {
        self.le(&self.scalar_like(c)?)
    }
}

fn binary_backward(
    op: BinaryOp,
    a: &Tensor,
    b: &Tensor,
    g: &Tensor,
) -> Result<smallvec::SmallVec<[Option<Tensor>; 2]>> {
    let (sa, sb) = (a.shape(), b.shape());
    let (need_a, need_b) = (a.requires_grad(), b.requires_grad());
    let side = |need: bool, f: &dyn Fn() -> Result<Tensor>, s: &Shape| -> Result<Option<Tensor>> {
        if need { Ok(Some(f()?.sum_to(s)?)) } else { Ok(None) }
    };
    Ok(match op {
        BinaryOp::Add => grads![side(need_a, &|| Ok(g.clone()), sa)?, side(need_b, &|| Ok(g.clone()), sb)?],
        BinaryOp::Sub => grads![side(need_a, &|| Ok(g.clone()), sa)?, side(need_b, &|| g.neg(), sb)?],
        BinaryOp::Mul => grads![side(need_a, &|| g.mul(b), sa)?, side(need_b, &|| g.mul(a), sb)?],
        BinaryOp::Div => {
            // d/db (a / b) = -(a / b) / b: dividing twice avoids squaring b (which under/overflows).
            grads![side(need_a, &|| g.div(b), sa)?, side(need_b, &|| g.mul(&a.div(b)?)?.div(b)?.neg(), sb)?]
        }
        BinaryOp::Pow => grads![
            side(
                need_a,
                &|| {
                    // d/da a^b = b a^(b-1), taken as 0 where b == 0 (a^0 is constant even at a = 0).
                    let d = g.mul(b)?.mul(&a.pow(&b.sub_scalar(1.0)?)?)?;
                    b.eq_scalar(0.0)?.where_cond(&a.scalar_like(0.0)?, &d)
                },
                sa
            )?,
            side(
                need_b,
                &|| {
                    // d/db a^b = a^b ln a, taken as 0 where a <= 0.
                    let pos = a.gt_scalar(0.0)?;
                    let safe = pos.where_cond(a, &a.scalar_like(1.0)?)?;
                    let ln = pos.where_cond(&safe.log()?, &a.scalar_like(0.0)?)?;
                    g.mul(&a.pow(b)?)?.mul(&ln)
                },
                sb
            )?
        ],
        BinaryOp::Atan2 => {
            // d/dy atan2(y, x) = x / r^2, d/dx = -y / r^2, with r^2 = x^2 + y^2; 0 at the origin.
            let r2 = a.square()?.add(&b.square()?)?;
            let zero = r2.eq_scalar(0.0)?;
            let safe = zero.where_cond(&a.scalar_like(1.0)?, &r2)?;
            let k = zero.where_cond(&a.scalar_like(0.0)?, &g.div(&safe)?)?;
            grads![side(need_a, &|| k.mul(b), sa)?, side(need_b, &|| k.mul(a)?.neg(), sb)?]
        }
        BinaryOp::Maximum | BinaryOp::Minimum => {
            let wins = if op == BinaryOp::Maximum { a.gt(b)? } else { a.lt(b)? };
            let ties = a.eq(b)?.mul_scalar(0.5)?;
            let wa = wins.add(&ties)?;
            grads![side(need_a, &|| g.mul(&wa), sa)?, side(need_b, &|| g.mul(&wa.rsub_scalar(1.0)?), sb)?]
        }
        _ => grads![None, None],
    })
}

#[cfg(test)]
mod tests {
    use crate::Tensor;

    fn t(v: &[f32], s: &[usize]) -> Tensor {
        Tensor::from_slice(v, s).unwrap()
    }

    #[test]
    fn broadcasting_binary() {
        let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = t(&[10.0, 20.0, 30.0], &[3]);
        assert_eq!(a.add(&b).unwrap().to_vec::<f32>().unwrap(), vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
        let c = t(&[1.0, 2.0], &[2, 1]);
        assert_eq!(a.mul(&c).unwrap().to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 8.0, 10.0, 12.0]);
        // Non-contiguous lhs.
        let at = a.t().unwrap(); // [3,2]
        let d = t(&[1.0, -1.0], &[2]);
        assert_eq!(at.mul(&d).unwrap().to_vec::<f32>().unwrap(), vec![1.0, -4.0, 2.0, -5.0, 3.0, -6.0]);
        let e = a.sub(&t(&[1.0, 2.0], &[2])).unwrap_err().to_string();
        assert!(e.contains("sub") && e.contains("[2, 3]") && e.contains("[2]"), "{e}");
        let outer = t(&[1.0, 2.0], &[2, 1]).add(&t(&[10.0, 20.0, 30.0], &[1, 3])).unwrap();
        assert_eq!(outer.dims(), &[2, 3]);
        assert_eq!(outer.to_vec::<f32>().unwrap(), vec![11.0, 21.0, 31.0, 12.0, 22.0, 32.0]);
    }

    #[test]
    fn unary_values() {
        let x = t(&[-2.0, -0.5, 0.0, 0.5, 2.0], &[5]);
        let close = |a: Vec<f32>, b: Vec<f32>| {
            for (x, y) in a.iter().zip(&b) {
                assert!((x - y).abs() < 1e-5, "{a:?} vs {b:?}");
            }
        };
        close(x.relu().unwrap().to_vec().unwrap(), vec![0.0, 0.0, 0.0, 0.5, 2.0]);
        close(x.leaky_relu(0.1).unwrap().to_vec().unwrap(), vec![-0.2, -0.05, 0.0, 0.5, 2.0]);
        let sig: Vec<f32> = [-2.0f32, -0.5, 0.0, 0.5, 2.0].iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect();
        close(x.sigmoid().unwrap().to_vec().unwrap(), sig.clone());
        let silu: Vec<f32> = [-2.0f32, -0.5, 0.0, 0.5, 2.0].iter().zip(&sig).map(|(v, s)| v * s).collect();
        close(x.silu().unwrap().to_vec().unwrap(), silu);
        close(
            x.hardswish().unwrap().to_vec().unwrap(),
            vec![-2.0 / 6.0, -0.5 * 2.5 / 6.0, 0.0, 0.5 * 3.5 / 6.0, 2.0 * 5.0 / 6.0],
        );
        close(
            x.softplus().unwrap().to_vec().unwrap(),
            [-2.0f32, -0.5, 0.0, 0.5, 2.0].iter().map(|v| v.exp().ln_1p()).collect(),
        );
        close(x.clamp(-1.0, 1.0).unwrap().to_vec().unwrap(), vec![-1.0, -0.5, 0.0, 0.5, 1.0]);
        close(
            x.abs().unwrap().sqrt().unwrap().to_vec().unwrap(),
            [2.0f32, 0.5, 0.0, 0.5, 2.0].iter().map(|v| v.sqrt()).collect(),
        );
        close(
            x.tanh().unwrap().to_vec().unwrap(),
            [-2.0f32, -0.5, 0.0, 0.5, 2.0].iter().map(|v| v.tanh()).collect(),
        );
        close(x.gt_scalar(0.0).unwrap().to_vec().unwrap(), vec![0.0, 0.0, 0.0, 1.0, 1.0]);
        close(x.le(&t(&[0.0], &[1])).unwrap().to_vec().unwrap(), vec![1.0, 1.0, 1.0, 0.0, 0.0]);
        close(
            x.mul_scalar(2.0).unwrap().add_scalar(1.0).unwrap().to_vec().unwrap(),
            vec![-3.0, 0.0, 1.0, 2.0, 5.0],
        );
        close(x.rsub_scalar(1.0).unwrap().to_vec().unwrap(), vec![3.0, 1.5, 1.0, 0.5, -1.0]);
        close(x.maximum(&t(&[0.25], &[])).unwrap().to_vec().unwrap(), vec![0.25, 0.25, 0.25, 0.5, 2.0]);
        close(x.square().unwrap().to_vec().unwrap(), vec![4.0, 0.25, 0.0, 0.25, 4.0]);
        close(x.exp().unwrap().log().unwrap().to_vec().unwrap(), vec![-2.0, -0.5, 0.0, 0.5, 2.0]);
        let v = [-2.0f32, -0.5, 0.0, 0.5, 2.0];
        close(x.atan().unwrap().to_vec().unwrap(), v.iter().map(|v| v.atan()).collect());
        close(
            x.mul_scalar(0.4).unwrap().asin().unwrap().to_vec().unwrap(),
            v.iter().map(|v| (v * 0.4).asin()).collect(),
        );
        close(
            x.mul_scalar(0.4).unwrap().acos().unwrap().to_vec().unwrap(),
            v.iter().map(|v| (v * 0.4).acos()).collect(),
        );
        close(x.floor().unwrap().to_vec().unwrap(), vec![-2.0, -1.0, 0.0, 0.0, 2.0]);
        close(x.ceil().unwrap().to_vec().unwrap(), vec![-2.0, -0.0, 0.0, 1.0, 2.0]);
        close(x.round().unwrap().to_vec().unwrap(), vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
        close(x.sign().unwrap().to_vec().unwrap(), vec![-1.0, -1.0, 0.0, 1.0, 1.0]);
        // erf(0.5) and erf(2) to 7 digits (Abramowitz & Stegun table 7.1).
        close(
            x.erf().unwrap().to_vec().unwrap(),
            vec![-0.995_322_3, -0.520_499_9, 0.0, 0.520_499_9, 0.995_322_3],
        );
        // atan2(y, x) over the four quadrants and the axes.
        let y = t(&[1.0, 1.0, -1.0, -1.0, 0.0, 2.0], &[6]);
        let xx = t(&[1.0, -1.0, -1.0, 1.0, -3.0, 0.0], &[6]);
        let want: Vec<f32> =
            [(1.0f32, 1.0f32), (1.0, -1.0), (-1.0, -1.0), (-1.0, 1.0), (0.0, -3.0), (2.0, 0.0)]
                .iter()
                .map(|(y, x)| y.atan2(*x))
                .collect();
        close(y.atan2(&xx).unwrap().to_vec().unwrap(), want);
        assert!(Tensor::from_slice(&[1u8], [1]).unwrap().exp().is_err());
    }

    #[test]
    fn large_parallel_paths_match_small() {
        let mut rng = crate::rng::Rng::seed(3);
        let a = Tensor::randn([64, 1024], 0.0, 1.0, &mut rng);
        let b = Tensor::randn([1024], 0.0, 1.0, &mut rng);
        let r = a.add(&b).unwrap().to_vec::<f32>().unwrap();
        let av = a.to_vec::<f32>().unwrap();
        let bv = b.to_vec::<f32>().unwrap();
        for (i, v) in r.iter().enumerate() {
            assert_eq!(*v, av[i] + bv[i % 1024]);
        }
        let s = a.t().unwrap().silu().unwrap().t().unwrap().to_vec::<f32>().unwrap();
        let s2 = a.silu().unwrap().to_vec::<f32>().unwrap();
        assert_eq!(s, s2);
    }
}
