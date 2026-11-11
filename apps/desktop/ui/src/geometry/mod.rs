// Adapted from GPUI (Apache-2.0): crates/gpui/src/geometry.rs
//! Geometry: units, lengths, points, sizes, bounds, edges and corners (GPUI's value types).

mod bounds;
mod corners;
mod edges;
mod length;
mod point;
mod size;
mod units;

pub use bounds::*;
pub use corners::*;
pub use edges::*;
pub use length::*;
pub use point::*;
pub use size::*;
pub use units::*;

use std::fmt::Debug;

/// Axis in a 2D cartesian space.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Axis {
    /// The y axis, or up and down
    Vertical,
    /// The x axis, or left and right
    Horizontal,
}

impl Axis {
    /// Swap this axis to the opposite axis.
    pub fn invert(self) -> Self {
        match self {
            Axis::Vertical => Axis::Horizontal,
            Axis::Horizontal => Axis::Vertical,
        }
    }
}

/// A trait for accessing the given unit along a certain axis.
pub trait Along {
    /// The unit associated with this type
    type Unit;

    /// Returns the unit along the given axis.
    fn along(&self, axis: Axis) -> Self::Unit;

    /// Applies the given function to the unit along the given axis and returns a new value.
    fn apply_along(&self, axis: Axis, f: impl FnOnce(Self::Unit) -> Self::Unit) -> Self;
}

/// Types that can be halved.
pub trait Half {
    /// Returns half of the current value.
    fn half(&self) -> Self;
}

impl Half for i32 {
    fn half(&self) -> Self {
        self / 2
    }
}

impl Half for f32 {
    fn half(&self) -> Self {
        self / 2.
    }
}

impl Half for DevicePixels {
    fn half(&self) -> Self {
        Self(self.0 / 2)
    }
}

impl Half for ScaledPixels {
    fn half(&self) -> Self {
        Self(self.0 / 2.)
    }
}

impl Half for Pixels {
    fn half(&self) -> Self {
        Self(self.0 / 2.)
    }
}

impl Half for Rems {
    fn half(&self) -> Self {
        Self(self.0 / 2.)
    }
}

/// Provides a trait for types that can negate their values.
pub trait Negate {
    /// Returns the negation of the given value
    fn negate(self) -> Self;
}

impl Negate for i32 {
    fn negate(self) -> Self {
        -self
    }
}

impl Negate for f32 {
    fn negate(self) -> Self {
        -self
    }
}

impl Negate for DevicePixels {
    fn negate(self) -> Self {
        Self(-self.0)
    }
}

impl Negate for ScaledPixels {
    fn negate(self) -> Self {
        Self(-self.0)
    }
}

impl Negate for Pixels {
    fn negate(self) -> Self {
        Self(-self.0)
    }
}

impl Negate for Rems {
    fn negate(self) -> Self {
        Self(-self.0)
    }
}

/// A trait for checking if a value is zero.
pub trait IsZero {
    /// Determines if the value is zero.
    fn is_zero(&self) -> bool;
}

impl IsZero for DevicePixels {
    fn is_zero(&self) -> bool {
        self.0 == 0
    }
}

impl IsZero for ScaledPixels {
    fn is_zero(&self) -> bool {
        self.0 == 0.
    }
}

impl IsZero for Pixels {
    fn is_zero(&self) -> bool {
        self.0 == 0.
    }
}

impl IsZero for Rems {
    fn is_zero(&self) -> bool {
        self.0 == 0.
    }
}

impl IsZero for AbsoluteLength {
    fn is_zero(&self) -> bool {
        match self {
            AbsoluteLength::Pixels(pixels) => pixels.is_zero(),
            AbsoluteLength::Rems(rems) => rems.is_zero(),
        }
    }
}

impl IsZero for DefiniteLength {
    fn is_zero(&self) -> bool {
        match self {
            DefiniteLength::Absolute(length) => length.is_zero(),
            DefiniteLength::Fraction(fraction) => *fraction == 0.,
        }
    }
}

impl IsZero for Length {
    fn is_zero(&self) -> bool {
        match self {
            Length::Definite(length) => length.is_zero(),
            Length::Auto => false,
        }
    }
}

impl<T: IsZero + Clone + Debug + Default + PartialEq> IsZero for Point<T> {
    fn is_zero(&self) -> bool {
        self.x.is_zero() && self.y.is_zero()
    }
}

impl<T> IsZero for Size<T>
where
    T: IsZero + Clone + Debug + Default + PartialEq,
{
    fn is_zero(&self) -> bool {
        self.width.is_zero() || self.height.is_zero()
    }
}

impl<T: IsZero + Clone + Debug + Default + PartialEq> IsZero for Bounds<T> {
    fn is_zero(&self) -> bool {
        self.size.is_zero()
    }
}

impl<T> IsZero for Corners<T>
where
    T: IsZero + Clone + Debug + Default + PartialEq,
{
    fn is_zero(&self) -> bool {
        self.top_left.is_zero()
            && self.top_right.is_zero()
            && self.bottom_right.is_zero()
            && self.bottom_left.is_zero()
    }
}

#[cfg(test)]
mod tests;
