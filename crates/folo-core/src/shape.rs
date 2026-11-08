//! Shapes, dimension arguments and broadcasting rules.

use smallvec::SmallVec;

use crate::{Error, Result};

/// Inline storage for dimension lists: tensors up to rank 4 never allocate for their shape.
pub type DimVec = SmallVec<[usize; 4]>;

/// The extent of each dimension of a tensor, outermost first. A rank-0 shape (`[]`) is a scalar.
#[derive(Clone, PartialEq, Eq, Hash, Default)]
pub struct Shape(DimVec);

impl Shape {
    /// Shape from a dimension slice.
    pub fn new(dims: &[usize]) -> Self {
        Shape(DimVec::from_slice(dims))
    }

    /// The rank-0 (scalar) shape.
    pub fn scalar() -> Self {
        Shape(DimVec::new())
    }

    /// The dimensions, outermost first.
    #[inline]
    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    /// Number of dimensions.
    #[inline]
    pub fn rank(&self) -> usize {
        self.0.len()
    }

    /// Total number of elements (1 for a scalar).
    #[inline]
    pub fn numel(&self) -> usize {
        self.0.iter().product()
    }

    /// Size of dimension `d` (negative counts from the end).
    pub fn dim(&self, d: impl Dim) -> Result<usize> {
        Ok(self.0[d.to_index(self.rank(), "dim")?])
    }

    /// Row-major (C-contiguous) strides in elements.
    pub fn contiguous_strides(&self) -> DimVec {
        let mut strides = DimVec::from_elem(0, self.rank());
        let mut acc = 1;
        for i in (0..self.rank()).rev() {
            strides[i] = acc;
            acc *= self.0[i];
        }
        strides
    }

    /// The NumPy/PyTorch broadcast of two shapes: align on the right, sizes must match or be 1.
    pub fn broadcast(&self, other: &Shape, op: &'static str) -> Result<Shape> {
        let r = self.rank().max(other.rank());
        let mut out = DimVec::from_elem(1, r);
        for (i, o) in out.iter_mut().enumerate() {
            let a = if i < r - self.rank() { 1 } else { self.0[i - (r - self.rank())] };
            let b = if i < r - other.rank() { 1 } else { other.0[i - (r - other.rank())] };
            *o = if a == b || b == 1 {
                a
            } else if a == 1 {
                b
            } else {
                return Err(Error::ShapeMismatch { op, lhs: self.clone(), rhs: other.clone() });
            };
        }
        Ok(Shape(out))
    }

    /// True if `self` can be broadcast to `target` (without changing `target`).
    pub fn broadcasts_to(&self, target: &Shape) -> bool {
        if self.rank() > target.rank() {
            return false;
        }
        let off = target.rank() - self.rank();
        self.0.iter().zip(&target.0[off..]).all(|(&a, &b)| a == b || a == 1)
    }

    pub(crate) fn dims_mut(&mut self) -> &mut DimVec {
        &mut self.0
    }

    /// The shape with dimension `d` set to `size`.
    pub(crate) fn with_dim(&self, d: usize, size: usize) -> Shape {
        let mut v = self.0.clone();
        v[d] = size;
        Shape(v)
    }
}

impl std::fmt::Debug for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.0.as_slice())
    }
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.0.as_slice())
    }
}

impl std::ops::Index<usize> for Shape {
    type Output = usize;
    fn index(&self, i: usize) -> &usize {
        &self.0[i]
    }
}

impl From<&[usize]> for Shape {
    fn from(d: &[usize]) -> Self {
        Shape::new(d)
    }
}
impl From<Vec<usize>> for Shape {
    fn from(d: Vec<usize>) -> Self {
        Shape(DimVec::from_vec(d))
    }
}
impl From<DimVec> for Shape {
    fn from(d: DimVec) -> Self {
        Shape(d)
    }
}
impl<const N: usize> From<[usize; N]> for Shape {
    fn from(d: [usize; N]) -> Self {
        Shape::new(&d)
    }
}
impl<const N: usize> From<&[usize; N]> for Shape {
    fn from(d: &[usize; N]) -> Self {
        Shape::new(d)
    }
}
impl From<&Shape> for Shape {
    fn from(s: &Shape) -> Self {
        s.clone()
    }
}
impl From<()> for Shape {
    fn from(_: ()) -> Self {
        Shape::scalar()
    }
}
impl From<usize> for Shape {
    fn from(d: usize) -> Self {
        Shape::new(&[d])
    }
}
impl From<(usize, usize)> for Shape {
    fn from(d: (usize, usize)) -> Self {
        Shape::new(&[d.0, d.1])
    }
}
impl From<(usize, usize, usize)> for Shape {
    fn from(d: (usize, usize, usize)) -> Self {
        Shape::new(&[d.0, d.1, d.2])
    }
}
impl From<(usize, usize, usize, usize)> for Shape {
    fn from(d: (usize, usize, usize, usize)) -> Self {
        Shape::new(&[d.0, d.1, d.2, d.3])
    }
}

/// A single dimension argument. Implemented for `usize` and signed integers; negative values count from the end
/// (`-1` is the last dimension), as in PyTorch.
pub trait Dim: Copy {
    /// Resolve against a tensor of rank `rank`.
    fn to_index(self, rank: usize, op: &'static str) -> Result<usize>;
}

fn signed_dim(d: i64, rank: usize, op: &'static str) -> Result<usize> {
    let r = rank as i64;
    let idx = if d < 0 { d + r } else { d };
    if idx < 0 || idx >= r {
        return Err(Error::DimOutOfRange { op, dim: d as isize, rank });
    }
    Ok(idx as usize)
}

impl Dim for usize {
    fn to_index(self, rank: usize, op: &'static str) -> Result<usize> {
        if self >= rank {
            return Err(Error::DimOutOfRange { op, dim: self as isize, rank });
        }
        Ok(self)
    }
}
macro_rules! signed_dim_impl {
    ($($t:ty),*) => {$(
        impl Dim for $t {
            fn to_index(self, rank: usize, op: &'static str) -> Result<usize> {
                signed_dim(self as i64, rank, op)
            }
        }
    )*};
}
signed_dim_impl!(i32, i64, isize);

/// A set of dimension arguments, for reductions: a single [`Dim`], an array, a slice, a `Vec`, or [`AllDims`].
pub trait Dims {
    /// Resolve against rank `rank`; returns sorted indices, rejecting duplicates.
    fn to_indices(&self, rank: usize, op: &'static str) -> Result<DimVec>;
}

fn finish_dims(mut v: DimVec, op: &'static str) -> Result<DimVec> {
    v.sort_unstable();
    let n = v.len();
    v.dedup();
    if v.len() != n {
        return Err(Error::invalid_arg(op, format!("repeated dimension in {v:?}")));
    }
    Ok(v)
}

macro_rules! dims_impl {
    ($($t:ty),*) => {$(
        impl Dims for $t {
            fn to_indices(&self, rank: usize, op: &'static str) -> Result<DimVec> {
                Ok(smallvec::smallvec![self.to_index(rank, op)?])
            }
        }
        impl<const N: usize> Dims for [$t; N] {
            fn to_indices(&self, rank: usize, op: &'static str) -> Result<DimVec> {
                self.as_slice().to_indices(rank, op)
            }
        }
        impl<const N: usize> Dims for &[$t; N] {
            fn to_indices(&self, rank: usize, op: &'static str) -> Result<DimVec> {
                self.as_slice().to_indices(rank, op)
            }
        }
        impl Dims for &[$t] {
            fn to_indices(&self, rank: usize, op: &'static str) -> Result<DimVec> {
                let v = self.iter().map(|d| d.to_index(rank, op)).collect::<Result<DimVec>>()?;
                finish_dims(v, op)
            }
        }
        impl Dims for Vec<$t> {
            fn to_indices(&self, rank: usize, op: &'static str) -> Result<DimVec> {
                self.as_slice().to_indices(rank, op)
            }
        }
    )*};
}
dims_impl!(usize, i32, i64, isize);

/// Every dimension, for reductions over the whole tensor: `t.sum(AllDims, false)`.
#[derive(Debug, Clone, Copy)]
pub struct AllDims;

impl Dims for AllDims {
    fn to_indices(&self, rank: usize, _op: &'static str) -> Result<DimVec> {
        Ok((0..rank).collect())
    }
}

/// A target shape for [`crate::Tensor::reshape`]: like a shape, but one entry may be `-1` (inferred).
/// Implemented for arrays/slices/`Vec`s of `usize` or signed integers, [`Shape`] and `()`.
pub trait ShapeSpec {
    /// Resolve against an element count.
    fn resolve(&self, numel: usize, op: &'static str) -> Result<Shape>;
}

fn resolve_signed(spec: &[i64], numel: usize, op: &'static str) -> Result<Shape> {
    let mut infer = None;
    let mut known = 1usize;
    let mut dims = DimVec::with_capacity(spec.len());
    for (i, &d) in spec.iter().enumerate() {
        if d == -1 {
            if infer.replace(i).is_some() {
                return Err(Error::invalid_arg(op, format!("more than one -1 in {spec:?}")));
            }
            dims.push(1);
        } else if d < 0 {
            return Err(Error::invalid_arg(op, format!("negative dimension in {spec:?}")));
        } else {
            known *= d as usize;
            dims.push(d as usize);
        }
    }
    if let Some(i) = infer {
        if known == 0 || !numel.is_multiple_of(known) {
            return Err(Error::invalid_arg(op, format!("cannot infer -1 in {spec:?} for {numel} elements")));
        }
        dims[i] = numel / known;
    } else if known != numel {
        return Err(Error::invalid_arg(op, format!("shape {spec:?} does not hold {numel} elements")));
    }
    Ok(Shape(dims))
}

impl ShapeSpec for Shape {
    fn resolve(&self, numel: usize, op: &'static str) -> Result<Shape> {
        if self.numel() != numel {
            return Err(Error::invalid_shape(op, self, format!("does not hold {numel} elements")));
        }
        Ok(self.clone())
    }
}
impl ShapeSpec for &Shape {
    fn resolve(&self, numel: usize, op: &'static str) -> Result<Shape> {
        (*self).resolve(numel, op)
    }
}
impl ShapeSpec for () {
    fn resolve(&self, numel: usize, op: &'static str) -> Result<Shape> {
        Shape::scalar().resolve(numel, op)
    }
}
macro_rules! shape_spec_impl {
    ($($t:ty),*) => {$(
        impl ShapeSpec for &[$t] {
            fn resolve(&self, numel: usize, op: &'static str) -> Result<Shape> {
                let v: SmallVec<[i64; 6]> = self.iter().map(|&d| d as i64).collect();
                resolve_signed(&v, numel, op)
            }
        }
        impl<const N: usize> ShapeSpec for [$t; N] {
            fn resolve(&self, numel: usize, op: &'static str) -> Result<Shape> {
                self.as_slice().resolve(numel, op)
            }
        }
        impl<const N: usize> ShapeSpec for &[$t; N] {
            fn resolve(&self, numel: usize, op: &'static str) -> Result<Shape> {
                self.as_slice().resolve(numel, op)
            }
        }
        impl ShapeSpec for Vec<$t> {
            fn resolve(&self, numel: usize, op: &'static str) -> Result<Shape> {
                self.as_slice().resolve(numel, op)
            }
        }
    )*};
}
shape_spec_impl!(usize, i32, i64, isize);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn broadcast_rules() {
        let a = Shape::new(&[8, 1, 6, 1]);
        let b = Shape::new(&[7, 1, 5]);
        assert_eq!(a.broadcast(&b, "t").unwrap().dims(), &[8, 7, 6, 5]);
        assert!(Shape::new(&[3]).broadcast(&Shape::new(&[4]), "t").is_err());
        let msg = Shape::new(&[2, 3]).broadcast(&Shape::new(&[4, 3]), "add").unwrap_err().to_string();
        assert!(msg.contains("add") && msg.contains("[2, 3]") && msg.contains("[4, 3]"), "{msg}");
        assert!(Shape::new(&[1, 3]).broadcasts_to(&Shape::new(&[2, 2, 3])));
        assert!(!Shape::new(&[2, 3]).broadcasts_to(&Shape::new(&[3, 3])));
    }

    #[test]
    fn strides_and_numel() {
        let s = Shape::new(&[2, 3, 4]);
        assert_eq!(s.numel(), 24);
        assert_eq!(s.contiguous_strides().as_slice(), &[12, 4, 1]);
        assert_eq!(Shape::scalar().numel(), 1);
        assert_eq!(s.dim(-1).unwrap(), 4);
        assert!(s.dim(3).is_err());
    }

    #[test]
    fn reshape_spec() {
        assert_eq!([2, -1].resolve(12, "r").unwrap().dims(), &[2, 6]);
        assert!([5, -1].resolve(12, "r").is_err());
        assert!([-1, -1].resolve(12, "r").is_err());
        assert_eq!([3usize, 4].resolve(12, "r").unwrap().dims(), &[3, 4]);
        assert!([1, 1].to_indices(2, "s").is_err());
        assert_eq!([-1, 0].to_indices(3, "s").unwrap().as_slice(), &[0, 2]);
    }
}
