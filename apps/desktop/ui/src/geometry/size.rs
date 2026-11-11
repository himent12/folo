// Adapted from GPUI (Apache-2.0): crates/gpui/src/geometry.rs
//! `Size`: a width and a height.

use super::*;
use std::fmt::{self, Debug, Display};
use std::ops::{Add, Div, Mul, MulAssign, Sub};

/// A size: a width and a height.
#[derive(Default, Clone, Copy, PartialEq, Hash)]
#[repr(C)]
pub struct Size<T: Clone + Debug + Default + PartialEq> {
    /// The width component of the size.
    pub width: T,
    /// The height component of the size.
    pub height: T,
}

impl<T: Clone + Debug + Default + PartialEq> Size<T> {
    /// Create a new Size, a synonym for [`size`]
    pub fn new(width: T, height: T) -> Self {
        size(width, height)
    }
}

/// Constructs a new `Size<T>` with the provided width and height.
pub const fn size<T>(width: T, height: T) -> Size<T>
where
    T: Clone + Debug + Default + PartialEq,
{
    Size { width, height }
}

impl<T> Size<T>
where
    T: Clone + Debug + Default + PartialEq,
{
    /// Applies a function to the width and height of the size, producing a new `Size<U>`.
    pub fn map<U>(&self, f: impl Fn(T) -> U) -> Size<U>
    where
        U: Clone + Debug + Default + PartialEq,
    {
        Size { width: f(self.width.clone()), height: f(self.height.clone()) }
    }
}

impl<T> Size<T>
where
    T: Clone + Debug + Default + PartialEq + Half,
{
    /// Compute the center point of the size.g
    pub fn center(&self) -> Point<T> {
        Point { x: self.width.half(), y: self.height.half() }
    }
}

impl Size<Pixels> {
    /// Scales the size by a given factor.
    pub fn scale(&self, factor: f32) -> Size<ScaledPixels> {
        Size { width: self.width.scale(factor), height: self.height.scale(factor) }
    }
}

impl<T> Along for Size<T>
where
    T: Clone + Debug + Default + PartialEq,
{
    type Unit = T;

    fn along(&self, axis: Axis) -> T {
        match axis {
            Axis::Horizontal => self.width.clone(),
            Axis::Vertical => self.height.clone(),
        }
    }

    /// Returns the value of this size along the given axis.
    fn apply_along(&self, axis: Axis, f: impl FnOnce(T) -> T) -> Self {
        match axis {
            Axis::Horizontal => Size { width: f(self.width.clone()), height: self.height.clone() },
            Axis::Vertical => Size { width: self.width.clone(), height: f(self.height.clone()) },
        }
    }
}

impl<T> Size<T>
where
    T: PartialOrd + Clone + Debug + Default + PartialEq,
{
    /// Returns a new `Size` with the maximum width and height from `self` and `other`.
    pub fn max(&self, other: &Self) -> Self {
        Size {
            width: if self.width >= other.width { self.width.clone() } else { other.width.clone() },
            height: if self.height >= other.height { self.height.clone() } else { other.height.clone() },
        }
    }

    /// Returns a new `Size` with the minimum width and height from `self` and `other`.
    pub fn min(&self, other: &Self) -> Self {
        Size {
            width: if self.width >= other.width { other.width.clone() } else { self.width.clone() },
            height: if self.height >= other.height { other.height.clone() } else { self.height.clone() },
        }
    }
}

impl<T> Sub for Size<T>
where
    T: Sub<Output = T> + Clone + Debug + Default + PartialEq,
{
    type Output = Size<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        Size { width: self.width - rhs.width, height: self.height - rhs.height }
    }
}

impl<T> Add for Size<T>
where
    T: Add<Output = T> + Clone + Debug + Default + PartialEq,
{
    type Output = Size<T>;

    fn add(self, rhs: Self) -> Self::Output {
        Size { width: self.width + rhs.width, height: self.height + rhs.height }
    }
}

impl<T, Rhs> Mul<Rhs> for Size<T>
where
    T: Mul<Rhs, Output = Rhs> + Clone + Debug + Default + PartialEq,
    Rhs: Clone + Debug + Default + PartialEq,
{
    type Output = Size<Rhs>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        Size { width: self.width * rhs.clone(), height: self.height * rhs }
    }
}

impl<T, S> MulAssign<S> for Size<T>
where
    T: Mul<S, Output = T> + Clone + Debug + Default + PartialEq,
    S: Clone,
{
    fn mul_assign(&mut self, rhs: S) {
        self.width = self.width.clone() * rhs.clone();
        self.height = self.height.clone() * rhs;
    }
}

impl<T> Eq for Size<T> where T: Eq + Clone + Debug + Default + PartialEq {}

impl<T> Debug for Size<T>
where
    T: Clone + Debug + Default + PartialEq,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Size {{ {:?} × {:?} }}", self.width, self.height)
    }
}

impl<T: Clone + Debug + Default + PartialEq + Display> Display for Size<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} × {}", self.width, self.height)
    }
}

impl<T: Clone + Debug + Default + PartialEq> From<Point<T>> for Size<T> {
    fn from(point: Point<T>) -> Self {
        Self { width: point.x, height: point.y }
    }
}

impl From<Size<Pixels>> for Size<DefiniteLength> {
    fn from(size: Size<Pixels>) -> Self {
        Size { width: size.width.into(), height: size.height.into() }
    }
}

impl From<Size<Pixels>> for Size<AbsoluteLength> {
    fn from(size: Size<Pixels>) -> Self {
        Size { width: size.width.into(), height: size.height.into() }
    }
}

impl Size<Length> {
    /// Returns a `Size` with both width and height set to fill the available space.
    pub fn full() -> Self {
        Self { width: relative(1.).into(), height: relative(1.).into() }
    }
}

impl Size<Length> {
    /// Returns a `Size` with both width and height set to `auto`, which allows the layout engine to determine the size.
    pub fn auto() -> Self {
        Self { width: Length::Auto, height: Length::Auto }
    }
}

impl<T, Rhs> Div<Rhs> for Size<T>
where
    T: Div<Rhs, Output = T> + Clone + Debug + Default + PartialEq,
    Rhs: Clone,
{
    type Output = Self;
    fn div(self, rhs: Rhs) -> Self {
        Size { width: self.width / rhs.clone(), height: self.height / rhs }
    }
}
