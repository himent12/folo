// Adapted from GPUI (Apache-2.0): crates/gpui/src/geometry.rs
//! `Edges`: a value for each side of a box.

use super::*;
use std::fmt::Debug;
use std::ops::{Mul, MulAssign};

/// A value for each edge of a box (top, right, bottom, left).
#[derive(Clone, Default, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct Edges<T: Clone + Debug + Default + PartialEq> {
    /// The size of the top edge.
    pub top: T,
    /// The size of the right edge.
    pub right: T,
    /// The size of the bottom edge.
    pub bottom: T,
    /// The size of the left edge.
    pub left: T,
}

impl<T> Mul for Edges<T>
where
    T: Mul<Output = T> + Clone + Debug + Default + PartialEq,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            top: self.top.clone() * rhs.top,
            right: self.right.clone() * rhs.right,
            bottom: self.bottom.clone() * rhs.bottom,
            left: self.left * rhs.left,
        }
    }
}

impl<T, S> MulAssign<S> for Edges<T>
where
    T: Mul<S, Output = T> + Clone + Debug + Default + PartialEq,
    S: Clone,
{
    fn mul_assign(&mut self, rhs: S) {
        self.top = self.top.clone() * rhs.clone();
        self.right = self.right.clone() * rhs.clone();
        self.bottom = self.bottom.clone() * rhs.clone();
        self.left = self.left.clone() * rhs;
    }
}

impl<T: Clone + Debug + Default + PartialEq + Copy> Copy for Edges<T> {}

impl<T: Clone + Debug + Default + PartialEq> Edges<T> {
    /// Constructs `Edges` where all sides are set to the same specified value.
    pub fn all(value: T) -> Self {
        Self { top: value.clone(), right: value.clone(), bottom: value.clone(), left: value }
    }

    /// Applies a function to each field of the `Edges`, producing a new `Edges<U>`.
    pub fn map<U>(&self, f: impl Fn(&T) -> U) -> Edges<U>
    where
        U: Clone + Debug + Default + PartialEq,
    {
        Edges { top: f(&self.top), right: f(&self.right), bottom: f(&self.bottom), left: f(&self.left) }
    }

    /// Checks if any of the edges satisfy a given predicate.
    pub fn any<F: Fn(&T) -> bool>(&self, predicate: F) -> bool {
        predicate(&self.top) || predicate(&self.right) || predicate(&self.bottom) || predicate(&self.left)
    }
}

impl Edges<Length> {
    /// Sets the edges of the `Edges` struct to `auto`, which is a special value that allows the layout engine to automatically determine the size of the edges.
    pub fn auto() -> Self {
        Self { top: Length::Auto, right: Length::Auto, bottom: Length::Auto, left: Length::Auto }
    }

    /// Sets the edges of the `Edges` struct to zero, which means no size or thickness.
    pub fn zero() -> Self {
        Self { top: px(0.).into(), right: px(0.).into(), bottom: px(0.).into(), left: px(0.).into() }
    }
}

impl Edges<DefiniteLength> {
    /// Sets the edges of the `Edges` struct to zero, which means no size or thickness.
    pub fn zero() -> Self {
        Self { top: px(0.).into(), right: px(0.).into(), bottom: px(0.).into(), left: px(0.).into() }
    }

    /// Converts the `DefiniteLength` to `Pixels` based on the parent size and the REM size.
    pub fn to_pixels(self, parent_size: Size<AbsoluteLength>, rem_size: Pixels) -> Edges<Pixels> {
        Edges {
            top: self.top.to_pixels(parent_size.height, rem_size),
            right: self.right.to_pixels(parent_size.width, rem_size),
            bottom: self.bottom.to_pixels(parent_size.height, rem_size),
            left: self.left.to_pixels(parent_size.width, rem_size),
        }
    }
}

impl Edges<AbsoluteLength> {
    /// Sets the edges of the `Edges` struct to zero, which means no size or thickness.
    pub fn zero() -> Self {
        Self { top: px(0.).into(), right: px(0.).into(), bottom: px(0.).into(), left: px(0.).into() }
    }

    /// Converts the `AbsoluteLength` to `Pixels` based on the `rem_size`.
    pub fn to_pixels(self, rem_size: Pixels) -> Edges<Pixels> {
        Edges {
            top: self.top.to_pixels(rem_size),
            right: self.right.to_pixels(rem_size),
            bottom: self.bottom.to_pixels(rem_size),
            left: self.left.to_pixels(rem_size),
        }
    }
}

impl Edges<Pixels> {
    /// Scales the `Edges<Pixels>` by a given factor, returning `Edges<ScaledPixels>`.
    pub fn scale(&self, factor: f32) -> Edges<ScaledPixels> {
        Edges {
            top: self.top.scale(factor),
            right: self.right.scale(factor),
            bottom: self.bottom.scale(factor),
            left: self.left.scale(factor),
        }
    }

    /// Returns the maximum value of any edge.
    pub fn max(&self) -> Pixels {
        self.top.max(self.right).max(self.bottom).max(self.left)
    }
}

impl From<f32> for Edges<Pixels> {
    fn from(val: f32) -> Self {
        let val: Pixels = val.into();
        val.into()
    }
}

impl From<Pixels> for Edges<Pixels> {
    fn from(val: Pixels) -> Self {
        Edges { top: val, right: val, bottom: val, left: val }
    }
}
