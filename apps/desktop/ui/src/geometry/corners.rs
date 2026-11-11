// Adapted from GPUI (Apache-2.0): crates/gpui/src/geometry.rs
//! `Corner` and `Corners`: a value for each corner of a box.

use super::*;
use std::cmp;
use std::fmt::Debug;
use std::ops::{Div, Mul, MulAssign};

/// One of the four corners of a box.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Corner {
    /// The top left corner
    TopLeft,
    /// The top right corner
    TopRight,
    /// The bottom left corner
    BottomLeft,
    /// The bottom right corner
    BottomRight,
}

impl Corner {
    /// Returns the directly opposite corner.
    #[must_use]
    pub fn opposite_corner(self) -> Self {
        match self {
            Corner::TopLeft => Corner::BottomRight,
            Corner::TopRight => Corner::BottomLeft,
            Corner::BottomLeft => Corner::TopRight,
            Corner::BottomRight => Corner::TopLeft,
        }
    }

    /// Returns the corner across from this corner, moving along the specified axis.
    #[must_use]
    pub fn other_side_corner_along(self, axis: Axis) -> Self {
        match axis {
            Axis::Vertical => match self {
                Corner::TopLeft => Corner::BottomLeft,
                Corner::TopRight => Corner::BottomRight,
                Corner::BottomLeft => Corner::TopLeft,
                Corner::BottomRight => Corner::TopRight,
            },
            Axis::Horizontal => match self {
                Corner::TopLeft => Corner::TopRight,
                Corner::TopRight => Corner::TopLeft,
                Corner::BottomLeft => Corner::BottomRight,
                Corner::BottomRight => Corner::BottomLeft,
            },
        }
    }
}

/// Represents the corners of a box in a 2D space, such as border radius.
#[derive(Clone, Default, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct Corners<T: Clone + Debug + Default + PartialEq> {
    /// The value associated with the top left corner.
    pub top_left: T,
    /// The value associated with the top right corner.
    pub top_right: T,
    /// The value associated with the bottom right corner.
    pub bottom_right: T,
    /// The value associated with the bottom left corner.
    pub bottom_left: T,
}

impl<T> Corners<T>
where
    T: Clone + Debug + Default + PartialEq,
{
    /// Constructs `Corners` where all sides are set to the same specified value.
    pub fn all(value: T) -> Self {
        Self {
            top_left: value.clone(),
            top_right: value.clone(),
            bottom_right: value.clone(),
            bottom_left: value,
        }
    }

    /// Returns the requested corner.
    #[must_use]
    pub fn corner(&self, corner: Corner) -> T {
        match corner {
            Corner::TopLeft => self.top_left.clone(),
            Corner::TopRight => self.top_right.clone(),
            Corner::BottomLeft => self.bottom_left.clone(),
            Corner::BottomRight => self.bottom_right.clone(),
        }
    }
}

impl Corners<AbsoluteLength> {
    /// Converts the `AbsoluteLength` to `Pixels` based on the provided rem size.
    pub fn to_pixels(self, rem_size: Pixels) -> Corners<Pixels> {
        Corners {
            top_left: self.top_left.to_pixels(rem_size),
            top_right: self.top_right.to_pixels(rem_size),
            bottom_right: self.bottom_right.to_pixels(rem_size),
            bottom_left: self.bottom_left.to_pixels(rem_size),
        }
    }
}

impl Corners<Pixels> {
    /// Scales the `Corners<Pixels>` by a given factor, returning `Corners<ScaledPixels>`.
    #[must_use]
    pub fn scale(&self, factor: f32) -> Corners<ScaledPixels> {
        Corners {
            top_left: self.top_left.scale(factor),
            top_right: self.top_right.scale(factor),
            bottom_right: self.bottom_right.scale(factor),
            bottom_left: self.bottom_left.scale(factor),
        }
    }

    /// Returns the maximum value of any corner.
    #[must_use]
    pub fn max(&self) -> Pixels {
        self.top_left.max(self.top_right).max(self.bottom_right).max(self.bottom_left)
    }
}

impl<T: Div<f32, Output = T> + Ord + Clone + Debug + Default + PartialEq> Corners<T> {
    /// Clamps corner radii to be less than or equal to half the shortest side of a quad.
    #[must_use]
    pub fn clamp_radii_for_quad_size(self, size: Size<T>) -> Corners<T> {
        let max = cmp::min(size.width, size.height) / 2.;
        Corners {
            top_left: cmp::min(self.top_left, max.clone()),
            top_right: cmp::min(self.top_right, max.clone()),
            bottom_right: cmp::min(self.bottom_right, max.clone()),
            bottom_left: cmp::min(self.bottom_left, max),
        }
    }
}

impl<T: Clone + Debug + Default + PartialEq> Corners<T> {
    /// Applies a function to each field of the `Corners`, producing a new `Corners<U>`.
    #[must_use]
    pub fn map<U>(&self, f: impl Fn(&T) -> U) -> Corners<U>
    where
        U: Clone + Debug + Default + PartialEq,
    {
        Corners {
            top_left: f(&self.top_left),
            top_right: f(&self.top_right),
            bottom_right: f(&self.bottom_right),
            bottom_left: f(&self.bottom_left),
        }
    }
}

impl<T> Mul for Corners<T>
where
    T: Mul<Output = T> + Clone + Debug + Default + PartialEq,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            top_left: self.top_left.clone() * rhs.top_left,
            top_right: self.top_right.clone() * rhs.top_right,
            bottom_right: self.bottom_right.clone() * rhs.bottom_right,
            bottom_left: self.bottom_left * rhs.bottom_left,
        }
    }
}

impl<T, S> MulAssign<S> for Corners<T>
where
    T: Mul<S, Output = T> + Clone + Debug + Default + PartialEq,
    S: Clone,
{
    fn mul_assign(&mut self, rhs: S) {
        self.top_left = self.top_left.clone() * rhs.clone();
        self.top_right = self.top_right.clone() * rhs.clone();
        self.bottom_right = self.bottom_right.clone() * rhs.clone();
        self.bottom_left = self.bottom_left.clone() * rhs;
    }
}

impl<T> Copy for Corners<T> where T: Copy + Clone + Debug + Default + PartialEq {}

impl From<f32> for Corners<Pixels> {
    fn from(val: f32) -> Self {
        Corners {
            top_left: val.into(),
            top_right: val.into(),
            bottom_right: val.into(),
            bottom_left: val.into(),
        }
    }
}

impl From<Pixels> for Corners<Pixels> {
    fn from(val: Pixels) -> Self {
        Corners { top_left: val, top_right: val, bottom_right: val, bottom_left: val }
    }
}
