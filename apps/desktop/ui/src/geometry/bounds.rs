// Adapted from GPUI (Apache-2.0): crates/gpui/src/geometry.rs
//! `Bounds`: a rectangle as an origin and a size.

use super::*;
use std::fmt::{self, Debug, Display};
use std::ops::{Add, Div, Mul, MulAssign, Neg, Sub};

/// A rectangle: an origin point and a size.
#[derive(Clone, Default, Debug, Eq, PartialEq, Hash)]
#[repr(C)]
pub struct Bounds<T: Clone + Debug + Default + PartialEq> {
    /// The origin point of this area.
    pub origin: Point<T>,
    /// The size of the rectangle.
    pub size: Size<T>,
}

/// Create a bounds with the given origin and size
pub fn bounds<T: Clone + Debug + Default + PartialEq>(origin: Point<T>, size: Size<T>) -> Bounds<T> {
    Bounds { origin, size }
}

impl<T> Bounds<T>
where
    T: Clone + Debug + Default + PartialEq,
{
    /// Creates a new `Bounds` with the specified origin and size.
    pub fn new(origin: Point<T>, size: Size<T>) -> Self {
        Bounds { origin, size }
    }
}

impl<T> Bounds<T>
where
    T: Sub<Output = T> + Clone + Debug + Default + PartialEq,
{
    /// Constructs a `Bounds` from two corner points: the top left and bottom right corners.
    pub fn from_corners(top_left: Point<T>, bottom_right: Point<T>) -> Self {
        let origin = Point { x: top_left.x.clone(), y: top_left.y.clone() };
        let size = Size { width: bottom_right.x - top_left.x, height: bottom_right.y - top_left.y };
        Bounds { origin, size }
    }

    /// Constructs a `Bounds` from a corner point and size. The specified corner will be placed at
    /// the specified origin.
    pub fn from_corner_and_size(corner: Corner, origin: Point<T>, size: Size<T>) -> Bounds<T> {
        let origin = match corner {
            Corner::TopLeft => origin,
            Corner::TopRight => Point { x: origin.x - size.width.clone(), y: origin.y },
            Corner::BottomLeft => Point { x: origin.x, y: origin.y - size.height.clone() },
            Corner::BottomRight => {
                Point { x: origin.x - size.width.clone(), y: origin.y - size.height.clone() }
            }
        };

        Bounds { origin, size }
    }
}

impl<T> Bounds<T>
where
    T: Sub<T, Output = T> + Half + Clone + Debug + Default + PartialEq,
{
    /// Creates a new bounds centered at the given point.
    pub fn centered_at(center: Point<T>, size: Size<T>) -> Self {
        let origin = Point { x: center.x - size.width.half(), y: center.y - size.height.half() };
        Self::new(origin, size)
    }
}

impl<T> Bounds<T>
where
    T: PartialOrd + Add<T, Output = T> + Clone + Debug + Default + PartialEq,
{
    /// Checks if this `Bounds` intersects with another `Bounds`.
    pub fn intersects(&self, other: &Bounds<T>) -> bool {
        let my_lower_right = self.bottom_right();
        let their_lower_right = other.bottom_right();

        self.origin.x < their_lower_right.x
            && my_lower_right.x > other.origin.x
            && self.origin.y < their_lower_right.y
            && my_lower_right.y > other.origin.y
    }
}

impl<T> Bounds<T>
where
    T: Add<T, Output = T> + Half + Clone + Debug + Default + PartialEq,
{
    /// Returns the center point of the bounds.
    pub fn center(&self) -> Point<T> {
        Point {
            x: self.origin.x.clone() + self.size.width.clone().half(),
            y: self.origin.y.clone() + self.size.height.clone().half(),
        }
    }
}

impl<T> Bounds<T>
where
    T: Add<T, Output = T> + Clone + Debug + Default + PartialEq,
{
    /// Calculates the half perimeter of a rectangle defined by the bounds.
    pub fn half_perimeter(&self) -> T {
        self.size.width.clone() + self.size.height.clone()
    }
}

impl<T> Bounds<T>
where
    T: Add<T, Output = T> + Sub<Output = T> + Clone + Debug + Default + PartialEq,
{
    /// Dilates the bounds by a specified amount in all directions.
    #[must_use]
    pub fn dilate(&self, amount: T) -> Bounds<T> {
        let double_amount = amount.clone() + amount.clone();
        Bounds {
            origin: self.origin.clone() - point(amount.clone(), amount),
            size: self.size.clone() + size(double_amount.clone(), double_amount),
        }
    }

    /// Extends the bounds different amounts in each direction.
    #[must_use]
    pub fn extend(&self, amount: Edges<T>) -> Bounds<T> {
        Bounds {
            origin: self.origin.clone() - point(amount.left.clone(), amount.top.clone()),
            size: self.size.clone()
                + size(amount.left.clone() + amount.right.clone(), amount.top.clone() + amount.bottom),
        }
    }
}

impl<T> Bounds<T>
where
    T: Add<T, Output = T> + Sub<T, Output = T> + Neg<Output = T> + Clone + Debug + Default + PartialEq,
{
    /// Inset the bounds by a specified amount. Equivalent to `dilate` with the amount negated.
    pub fn inset(&self, amount: T) -> Self {
        self.dilate(-amount)
    }
}

impl<T: PartialOrd + Add<T, Output = T> + Sub<Output = T> + Clone + Debug + Default + PartialEq> Bounds<T> {
    /// Calculates the intersection of two `Bounds` objects.
    pub fn intersect(&self, other: &Self) -> Self {
        let upper_left = self.origin.max(&other.origin);
        let bottom_right = self.bottom_right().min(&other.bottom_right());
        Self::from_corners(upper_left, bottom_right)
    }

    /// Computes the union of two `Bounds`.
    pub fn union(&self, other: &Self) -> Self {
        let top_left = self.origin.min(&other.origin);
        let bottom_right = self.bottom_right().max(&other.bottom_right());
        Bounds::from_corners(top_left, bottom_right)
    }
}

impl<T> Bounds<T>
where
    T: Add<T, Output = T> + Sub<T, Output = T> + Clone + Debug + Default + PartialEq,
{
    /// Computes the space available within outer bounds.
    pub fn space_within(&self, outer: &Self) -> Edges<T> {
        Edges {
            top: self.top() - outer.top(),
            right: outer.right() - self.right(),
            bottom: outer.bottom() - self.bottom(),
            left: self.left() - outer.left(),
        }
    }
}

impl<T, Rhs> Mul<Rhs> for Bounds<T>
where
    T: Mul<Rhs, Output = Rhs> + Clone + Debug + Default + PartialEq,
    Point<T>: Mul<Rhs, Output = Point<Rhs>>,
    Rhs: Clone + Debug + Default + PartialEq,
{
    type Output = Bounds<Rhs>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        Bounds { origin: self.origin * rhs.clone(), size: self.size * rhs }
    }
}

impl<T, S> MulAssign<S> for Bounds<T>
where
    T: Mul<S, Output = T> + Clone + Debug + Default + PartialEq,
    S: Clone,
{
    fn mul_assign(&mut self, rhs: S) {
        self.origin *= rhs.clone();
        self.size *= rhs;
    }
}

impl<T, S> Div<S> for Bounds<T>
where
    Size<T>: Div<S, Output = Size<T>>,
    T: Div<S, Output = T> + Clone + Debug + Default + PartialEq,
    S: Clone,
{
    type Output = Self;

    fn div(self, rhs: S) -> Self {
        Self { origin: self.origin / rhs.clone(), size: self.size / rhs }
    }
}

impl<T> Add<Point<T>> for Bounds<T>
where
    T: Add<T, Output = T> + Clone + Debug + Default + PartialEq,
{
    type Output = Self;

    fn add(self, rhs: Point<T>) -> Self {
        Self { origin: self.origin + rhs, size: self.size }
    }
}

impl<T> Sub<Point<T>> for Bounds<T>
where
    T: Sub<T, Output = T> + Clone + Debug + Default + PartialEq,
{
    type Output = Self;

    fn sub(self, rhs: Point<T>) -> Self {
        Self { origin: self.origin - rhs, size: self.size }
    }
}

impl<T> Bounds<T>
where
    T: Add<T, Output = T> + Clone + Debug + Default + PartialEq,
{
    /// Returns the top edge of the bounds.
    pub fn top(&self) -> T {
        self.origin.y.clone()
    }

    /// Returns the bottom edge of the bounds.
    pub fn bottom(&self) -> T {
        self.origin.y.clone() + self.size.height.clone()
    }

    /// Returns the left edge of the bounds.
    pub fn left(&self) -> T {
        self.origin.x.clone()
    }

    /// Returns the right edge of the bounds.
    pub fn right(&self) -> T {
        self.origin.x.clone() + self.size.width.clone()
    }

    /// Returns the top right corner point of the bounds.
    pub fn top_right(&self) -> Point<T> {
        Point { x: self.origin.x.clone() + self.size.width.clone(), y: self.origin.y.clone() }
    }

    /// Returns the bottom right corner point of the bounds.
    pub fn bottom_right(&self) -> Point<T> {
        Point {
            x: self.origin.x.clone() + self.size.width.clone(),
            y: self.origin.y.clone() + self.size.height.clone(),
        }
    }

    /// Returns the bottom left corner point of the bounds.
    pub fn bottom_left(&self) -> Point<T> {
        Point { x: self.origin.x.clone(), y: self.origin.y.clone() + self.size.height.clone() }
    }

    /// Returns the requested corner point of the bounds.
    pub fn corner(&self, corner: Corner) -> Point<T> {
        match corner {
            Corner::TopLeft => self.origin.clone(),
            Corner::TopRight => self.top_right(),
            Corner::BottomLeft => self.bottom_left(),
            Corner::BottomRight => self.bottom_right(),
        }
    }
}

impl<T> Bounds<T>
where
    T: Add<T, Output = T> + PartialOrd + Clone + Debug + Default + PartialEq,
{
    /// Checks if the given point is within the bounds.
    pub fn contains(&self, point: &Point<T>) -> bool {
        point.x >= self.origin.x
            && point.x <= self.origin.x.clone() + self.size.width.clone()
            && point.y >= self.origin.y
            && point.y <= self.origin.y.clone() + self.size.height.clone()
    }

    /// Checks if this bounds is completely contained within another bounds.
    pub fn is_contained_within(&self, other: &Self) -> bool {
        other.contains(&self.origin) && other.contains(&self.bottom_right())
    }

    /// Applies a function to the origin and size of the bounds, producing a new `Bounds<U>`.
    pub fn map<U>(&self, f: impl Fn(T) -> U) -> Bounds<U>
    where
        U: Clone + Debug + Default + PartialEq,
    {
        Bounds { origin: self.origin.map(&f), size: self.size.map(f) }
    }

    /// Applies a function to the origin  of the bounds, producing a new `Bounds` with the new origin
    pub fn map_origin(self, f: impl Fn(T) -> T) -> Bounds<T> {
        Bounds { origin: self.origin.map(f), size: self.size }
    }

    /// Applies a function to the origin  of the bounds, producing a new `Bounds` with the new origin
    pub fn map_size(self, f: impl Fn(T) -> T) -> Bounds<T> {
        Bounds { origin: self.origin, size: self.size.map(f) }
    }
}

impl<T> Bounds<T>
where
    T: Add<T, Output = T> + Sub<T, Output = T> + PartialOrd + Clone + Debug + Default + PartialEq,
{
    /// Convert a point to the coordinate space defined by this Bounds
    pub fn localize(&self, point: &Point<T>) -> Option<Point<T>> {
        self.contains(point).then(|| point.relative_to(&self.origin))
    }
}

/// Checks if the bounds represent an empty area.
impl<T: PartialOrd + Clone + Debug + Default + PartialEq> Bounds<T> {
    /// Checks if the bounds represent an empty area.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.size.width <= T::default() || self.size.height <= T::default()
    }
}

impl<T: Clone + Debug + Default + PartialEq + Display + Add<T, Output = T>> Display for Bounds<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} - {} (size {})", self.origin, self.bottom_right(), self.size)
    }
}

impl Size<DevicePixels> {
    /// Converts the size from physical to logical pixels.
    pub(crate) fn to_pixels(self, scale_factor: f32) -> Size<Pixels> {
        size(px(self.width.0 as f32 / scale_factor), px(self.height.0 as f32 / scale_factor))
    }
}

impl Size<Pixels> {
    /// Converts the size from logical to physical pixels.
    pub(crate) fn to_device_pixels(self, scale_factor: f32) -> Size<DevicePixels> {
        size(
            DevicePixels((self.width.0 * scale_factor).round() as i32),
            DevicePixels((self.height.0 * scale_factor).round() as i32),
        )
    }
}

impl Bounds<Pixels> {
    /// Scales the bounds by a given factor, typically used to adjust for display scaling.
    pub fn scale(&self, factor: f32) -> Bounds<ScaledPixels> {
        Bounds { origin: self.origin.scale(factor), size: self.size.scale(factor) }
    }

    /// Convert the bounds from logical pixels to physical pixels
    pub fn to_device_pixels(self, factor: f32) -> Bounds<DevicePixels> {
        Bounds {
            origin: point(
                DevicePixels((self.origin.x.0 * factor).round() as i32),
                DevicePixels((self.origin.y.0 * factor).round() as i32),
            ),
            size: self.size.to_device_pixels(factor),
        }
    }
}

impl Bounds<DevicePixels> {
    /// Convert the bounds from physical pixels to logical pixels
    pub fn to_pixels(self, scale_factor: f32) -> Bounds<Pixels> {
        Bounds {
            origin: point(
                px(self.origin.x.0 as f32 / scale_factor),
                px(self.origin.y.0 as f32 / scale_factor),
            ),
            size: self.size.to_pixels(scale_factor),
        }
    }
}

impl<T: Copy + Clone + Debug + Default + PartialEq> Copy for Bounds<T> {}
