// Adapted from GPUI (Apache-2.0): crates/gpui/src/geometry.rs
//! Lengths: `Rems`, `AbsoluteLength`, `DefiniteLength`, `Length`.

use super::*;
use std::fmt::{self, Debug, Display};
use std::ops::{Add, Div, Mul, Neg, Sub};

/// A length in rems: a multiple of the root font size.
#[derive(Clone, Copy, Default, PartialEq)]
pub struct Rems(pub f32);

impl Rems {
    /// Convert this Rem value to pixels.
    pub fn to_pixels(self, rem_size: Pixels) -> Pixels {
        self * rem_size
    }
}

impl Mul<Pixels> for Rems {
    type Output = Pixels;

    fn mul(self, other: Pixels) -> Pixels {
        Pixels(self.0 * other.0)
    }
}

impl Display for Rems {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}rem", self.0)
    }
}

impl Debug for Rems {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Display::fmt(self, f)
    }
}

/// Represents an absolute length in pixels or rems.
#[derive(Clone, Copy, PartialEq)]
pub enum AbsoluteLength {
    /// A length in pixels.
    Pixels(Pixels),
    /// A length in rems.
    Rems(Rems),
}

impl AbsoluteLength {
    /// Checks if the absolute length is zero.
    pub fn is_zero(&self) -> bool {
        match self {
            AbsoluteLength::Pixels(px) => px.0 == 0.0,
            AbsoluteLength::Rems(rems) => rems.0 == 0.0,
        }
    }
}

impl From<Pixels> for AbsoluteLength {
    fn from(pixels: Pixels) -> Self {
        AbsoluteLength::Pixels(pixels)
    }
}

impl From<Rems> for AbsoluteLength {
    fn from(rems: Rems) -> Self {
        AbsoluteLength::Rems(rems)
    }
}

impl AbsoluteLength {
    /// Converts an `AbsoluteLength` to `Pixels` based on a given `rem_size`.
    pub fn to_pixels(self, rem_size: Pixels) -> Pixels {
        match self {
            AbsoluteLength::Pixels(pixels) => pixels,
            AbsoluteLength::Rems(rems) => rems.to_pixels(rem_size),
        }
    }

    /// Converts an `AbsoluteLength` to `Rems` based on a given `rem_size`.
    pub fn to_rems(self, rem_size: Pixels) -> Rems {
        match self {
            AbsoluteLength::Pixels(pixels) => Rems(pixels.0 / rem_size.0),
            AbsoluteLength::Rems(rems) => rems,
        }
    }
}

impl Default for AbsoluteLength {
    fn default() -> Self {
        px(0.).into()
    }
}

impl Display for AbsoluteLength {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pixels(pixels) => write!(f, "{pixels}"),
            Self::Rems(rems) => write!(f, "{rems}"),
        }
    }
}

impl Debug for AbsoluteLength {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Display::fmt(self, f)
    }
}

/// A non-auto length that can be defined in pixels, rems, or percent of parent.
#[derive(Clone, Copy, PartialEq)]
pub enum DefiniteLength {
    /// An absolute length specified in pixels or rems.
    Absolute(AbsoluteLength),
    /// A relative length specified as a fraction of the parent's size, between 0 and 1.
    Fraction(f32),
}

impl DefiniteLength {
    /// Converts the `DefiniteLength` to `Pixels` based on a given `base_size` and `rem_size`.
    pub fn to_pixels(self, base_size: AbsoluteLength, rem_size: Pixels) -> Pixels {
        match self {
            DefiniteLength::Absolute(size) => size.to_pixels(rem_size),
            DefiniteLength::Fraction(fraction) => match base_size {
                AbsoluteLength::Pixels(px) => px * fraction,
                AbsoluteLength::Rems(rems) => rems * rem_size * fraction,
            },
        }
    }
}

impl Debug for DefiniteLength {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Display::fmt(self, f)
    }
}

impl Display for DefiniteLength {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DefiniteLength::Absolute(length) => write!(f, "{length}"),
            DefiniteLength::Fraction(fraction) => write!(f, "{}%", (fraction * 100.0) as i32),
        }
    }
}

impl From<Pixels> for DefiniteLength {
    fn from(pixels: Pixels) -> Self {
        Self::Absolute(pixels.into())
    }
}

impl From<Rems> for DefiniteLength {
    fn from(rems: Rems) -> Self {
        Self::Absolute(rems.into())
    }
}

impl From<AbsoluteLength> for DefiniteLength {
    fn from(length: AbsoluteLength) -> Self {
        Self::Absolute(length)
    }
}

impl Default for DefiniteLength {
    fn default() -> Self {
        Self::Absolute(AbsoluteLength::default())
    }
}

/// A length that can be defined in pixels, rems, percent of parent, or auto.
#[derive(Clone, Copy, PartialEq)]
pub enum Length {
    /// A definite length specified either in pixels, rems, or as a fraction of the parent's size.
    Definite(DefiniteLength),
    /// An automatic length that is determined by the context in which it is used.
    Auto,
}

impl Debug for Length {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Display::fmt(self, f)
    }
}

impl Display for Length {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Length::Definite(definite_length) => write!(f, "{}", definite_length),
            Length::Auto => write!(f, "auto"),
        }
    }
}

/// Constructs a `DefiniteLength` representing a relative fraction of a parent size.
pub const fn relative(fraction: f32) -> DefiniteLength {
    DefiniteLength::Fraction(fraction)
}

/// Returns the Golden Ratio, i.e. `~(1.0 + sqrt(5.0)) / 2.0`.
pub fn phi() -> DefiniteLength {
    relative(1.618_034)
}

/// Constructs a `Rems` value representing a length in rems.
pub fn rems(rems: f32) -> Rems {
    Rems(rems)
}

/// Constructs a `Pixels` value representing a length in pixels.
pub const fn px(pixels: f32) -> Pixels {
    Pixels(pixels)
}

/// Returns a `Length` representing an automatic length.
pub fn auto() -> Length {
    Length::Auto
}

impl From<Pixels> for Length {
    fn from(pixels: Pixels) -> Self {
        Self::Definite(pixels.into())
    }
}

impl From<Rems> for Length {
    fn from(rems: Rems) -> Self {
        Self::Definite(rems.into())
    }
}

impl From<DefiniteLength> for Length {
    fn from(length: DefiniteLength) -> Self {
        Self::Definite(length)
    }
}

impl From<AbsoluteLength> for Length {
    fn from(length: AbsoluteLength) -> Self {
        Self::Definite(length.into())
    }
}

impl Default for Length {
    fn default() -> Self {
        Self::Definite(DefiniteLength::default())
    }
}

impl From<()> for Length {
    fn from(_: ()) -> Self {
        Self::Definite(DefiniteLength::default())
    }
}

impl Add for Rems {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Rems(self.0 + rhs.0)
    }
}

impl Sub for Rems {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Rems(self.0 - rhs.0)
    }
}

impl Mul<f32> for Rems {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self {
        Rems(self.0 * rhs)
    }
}

impl Div<f32> for Rems {
    type Output = Self;
    fn div(self, rhs: f32) -> Self {
        Rems(self.0 / rhs)
    }
}

impl Neg for Rems {
    type Output = Self;
    fn neg(self) -> Self {
        Rems(-self.0)
    }
}

impl Neg for AbsoluteLength {
    type Output = Self;
    fn neg(self) -> Self {
        match self {
            AbsoluteLength::Pixels(p) => AbsoluteLength::Pixels(-p),
            AbsoluteLength::Rems(r) => AbsoluteLength::Rems(-r),
        }
    }
}

impl Neg for DefiniteLength {
    type Output = Self;
    fn neg(self) -> Self {
        match self {
            DefiniteLength::Absolute(a) => DefiniteLength::Absolute(-a),
            DefiniteLength::Fraction(f) => DefiniteLength::Fraction(-f),
        }
    }
}
