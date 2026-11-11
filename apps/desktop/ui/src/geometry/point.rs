// Adapted from GPUI (Apache-2.0): crates/gpui/src/geometry.rs
//! `Point`: a location in 2D space.

use super::*;
use std::fmt::{self, Debug, Display};
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Sub, SubAssign};

/// A point in a 2D cartesian space.
#[derive(Default, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct Point<T: Clone + Debug + Default + PartialEq> {
    /// The x coordinate of the point.
    pub x: T,
    /// The y coordinate of the point.
    pub y: T,
}

/// Constructs a new `Point<T>` with the given x and y coordinates.
pub const fn point<T: Clone + Debug + Default + PartialEq>(x: T, y: T) -> Point<T> {
    Point { x, y }
}

impl<T: Clone + Debug + Default + PartialEq> Point<T> {
    /// Creates a new `Point` with the specified `x` and `y` coordinates.
    pub const fn new(x: T, y: T) -> Self {
        Self { x, y }
    }

    /// Transforms the point to a `Point<U>` by applying the given function to both coordinates.
    #[must_use]
    pub fn map<U: Clone + Debug + Default + PartialEq>(&self, f: impl Fn(T) -> U) -> Point<U> {
        Point { x: f(self.x.clone()), y: f(self.y.clone()) }
    }
}

impl<T: Clone + Debug + Default + PartialEq> Along for Point<T> {
    type Unit = T;

    fn along(&self, axis: Axis) -> T {
        match axis {
            Axis::Horizontal => self.x.clone(),
            Axis::Vertical => self.y.clone(),
        }
    }

    fn apply_along(&self, axis: Axis, f: impl FnOnce(T) -> T) -> Point<T> {
        match axis {
            Axis::Horizontal => Point { x: f(self.x.clone()), y: self.y.clone() },
            Axis::Vertical => Point { x: self.x.clone(), y: f(self.y.clone()) },
        }
    }
}

impl<T: Clone + Debug + Default + PartialEq + Negate> Negate for Point<T> {
    fn negate(self) -> Self {
        self.map(Negate::negate)
    }
}

impl Point<Pixels> {
    /// Scales the point by a given factor, which is typically derived from the resolution
    /// of a target display to ensure proper sizing of UI elements.
    pub fn scale(&self, factor: f32) -> Point<ScaledPixels> {
        Point { x: self.x.scale(factor), y: self.y.scale(factor) }
    }

    /// Calculates the Euclidean distance from the origin (0, 0) to this point.
    pub fn magnitude(&self) -> f64 {
        ((self.x.0.powi(2) + self.y.0.powi(2)) as f64).sqrt()
    }
}

impl<T> Point<T>
where
    T: Sub<T, Output = T> + Clone + Debug + Default + PartialEq,
{
    /// Get the position of this point, relative to the given origin
    pub fn relative_to(&self, origin: &Point<T>) -> Point<T> {
        point(self.x.clone() - origin.x.clone(), self.y.clone() - origin.y.clone())
    }
}

impl<T, Rhs> Mul<Rhs> for Point<T>
where
    T: Mul<Rhs, Output = T> + Clone + Debug + Default + PartialEq,
    Rhs: Clone + Debug,
{
    type Output = Point<T>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        Point { x: self.x * rhs.clone(), y: self.y * rhs }
    }
}

impl<T, S> MulAssign<S> for Point<T>
where
    T: Mul<S, Output = T> + Clone + Debug + Default + PartialEq,
    S: Clone,
{
    fn mul_assign(&mut self, rhs: S) {
        self.x = self.x.clone() * rhs.clone();
        self.y = self.y.clone() * rhs;
    }
}

impl<T, S> Div<S> for Point<T>
where
    T: Div<S, Output = T> + Clone + Debug + Default + PartialEq,
    S: Clone,
{
    type Output = Self;

    fn div(self, rhs: S) -> Self::Output {
        Self { x: self.x / rhs.clone(), y: self.y / rhs }
    }
}

impl<T> Point<T>
where
    T: PartialOrd + Clone + Debug + Default + PartialEq,
{
    /// Returns a new point with the maximum values of each dimension from `self` and `other`.
    pub fn max(&self, other: &Self) -> Self {
        Point {
            x: if self.x > other.x { self.x.clone() } else { other.x.clone() },
            y: if self.y > other.y { self.y.clone() } else { other.y.clone() },
        }
    }

    /// Returns a new point with the minimum values of each dimension from `self` and `other`.
    pub fn min(&self, other: &Self) -> Self {
        Point {
            x: if self.x <= other.x { self.x.clone() } else { other.x.clone() },
            y: if self.y <= other.y { self.y.clone() } else { other.y.clone() },
        }
    }

    /// Clamps the point to a specified range.
    pub fn clamp(&self, min: &Self, max: &Self) -> Self {
        self.max(min).min(max)
    }
}

impl<T: Clone + Debug + Default + PartialEq> Clone for Point<T> {
    fn clone(&self) -> Self {
        Self { x: self.x.clone(), y: self.y.clone() }
    }
}

impl<T: Clone + Debug + Default + PartialEq + Display> Display for Point<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

impl<T: Clone + Debug + Default + PartialEq + Add<Output = T>> Add for Point<T> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Point { x: self.x + rhs.x, y: self.y + rhs.y }
    }
}

impl<T: Clone + Debug + Default + PartialEq + AddAssign> AddAssign for Point<T> {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}

impl<T: Clone + Debug + Default + PartialEq + Sub<Output = T>> Sub for Point<T> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Point { x: self.x - rhs.x, y: self.y - rhs.y }
    }
}

impl<T: Clone + Debug + Default + PartialEq + SubAssign> SubAssign for Point<T> {
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
    }
}
