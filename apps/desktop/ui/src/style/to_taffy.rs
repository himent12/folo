// Adapted from GPUI (Apache-2.0): crates/gpui/src/taffy.rs (ToTaffy)
//! `Style::to_taffy`: GPUI's style-to-taffy mapping. Lengths are scaled to device pixels here (taffy
//! lays out in device pixels); nothing is rounded here (the layout engine rounds taffy's output).

use super::Style;
use crate::{AbsoluteLength, DefiniteLength, Edges, Length, Pixels, Point, Size};
use std::fmt::Debug;
use taffy::geometry::{Point as TaffyPoint, Rect as TaffyRect, Size as TaffySize};
use taffy::style::{Dimension, GridTemplateComponent, LengthPercentage, LengthPercentageAuto};

/// A value that converts to its taffy form at a rem size and scale factor.
pub trait ToTaffy<Output> {
    /// The taffy value, in device pixels.
    fn to_taffy(&self, rem_size: Pixels, scale_factor: f32) -> Output;
}

impl Style {
    /// This style as taffy's, with every length in device pixels (`rem_size` and `scale_factor` applied).
    pub fn to_taffy(&self, rem_size: Pixels, scale_factor: f32) -> taffy::style::Style {
        ToTaffy::to_taffy(self, rem_size, scale_factor)
    }
}

/// `grid-template-columns: repeat(<count>, minmax(0, 1fr))`, as GPUI's `grid_cols` / `grid_rows`.
fn to_grid_repeat(unit: &Option<u16>) -> Vec<GridTemplateComponent<String>> {
    use taffy::style_helpers::{fr, length, minmax, repeat};
    unit.map(|count| vec![repeat(count, vec![minmax(length(0.0_f32), fr(1.0_f32))])]).unwrap_or_default()
}

impl ToTaffy<taffy::style::Style> for Style {
    fn to_taffy(&self, rem_size: Pixels, scale_factor: f32) -> taffy::style::Style {
        taffy::style::Style {
            display: self.display.into(),
            overflow: self.overflow.into(),
            scrollbar_width: self.scrollbar_width.to_taffy(rem_size, scale_factor),
            position: self.position.into(),
            inset: self.inset.to_taffy(rem_size, scale_factor),
            size: self.size.to_taffy(rem_size, scale_factor),
            min_size: self.min_size.to_taffy(rem_size, scale_factor),
            max_size: self.max_size.to_taffy(rem_size, scale_factor),
            aspect_ratio: self.aspect_ratio,
            margin: self.margin.to_taffy(rem_size, scale_factor),
            padding: self.padding.to_taffy(rem_size, scale_factor),
            border: self.border_widths.to_taffy(rem_size, scale_factor),
            align_items: self.align_items.map(|x| x.into()),
            align_self: self.align_self.map(|x| x.into()),
            align_content: self.align_content.map(|x| x.into()),
            justify_content: self.justify_content.map(|x| x.into()),
            gap: self.gap.to_taffy(rem_size, scale_factor),
            flex_direction: self.flex_direction.into(),
            flex_wrap: self.flex_wrap.into(),
            flex_basis: self.flex_basis.to_taffy(rem_size, scale_factor),
            flex_grow: self.flex_grow,
            flex_shrink: self.flex_shrink,
            grid_template_rows: to_grid_repeat(&self.grid_rows),
            grid_template_columns: to_grid_repeat(&self.grid_cols),
            ..Default::default()
        }
    }
}

/// `pixels * scale_factor` for an absolute length (rems resolved first).
fn absolute_px(length: &AbsoluteLength, rem_size: Pixels, scale_factor: f32) -> f32 {
    match length {
        AbsoluteLength::Pixels(pixels) => f32::from(*pixels) * scale_factor,
        AbsoluteLength::Rems(rems) => f32::from(*rems * rem_size) * scale_factor,
    }
}

impl ToTaffy<f32> for AbsoluteLength {
    fn to_taffy(&self, rem_size: Pixels, scale_factor: f32) -> f32 {
        absolute_px(self, rem_size, scale_factor)
    }
}

impl ToTaffy<LengthPercentageAuto> for Length {
    fn to_taffy(&self, rem_size: Pixels, scale_factor: f32) -> LengthPercentageAuto {
        match self {
            Length::Definite(length) => length.to_taffy(rem_size, scale_factor),
            Length::Auto => LengthPercentageAuto::auto(),
        }
    }
}

impl ToTaffy<Dimension> for Length {
    fn to_taffy(&self, rem_size: Pixels, scale_factor: f32) -> Dimension {
        match self {
            Length::Definite(length) => length.to_taffy(rem_size, scale_factor),
            Length::Auto => Dimension::auto(),
        }
    }
}

impl ToTaffy<LengthPercentage> for DefiniteLength {
    fn to_taffy(&self, rem_size: Pixels, scale_factor: f32) -> LengthPercentage {
        match self {
            DefiniteLength::Absolute(length) => {
                LengthPercentage::length(absolute_px(length, rem_size, scale_factor))
            }
            DefiniteLength::Fraction(fraction) => LengthPercentage::percent(*fraction),
        }
    }
}

impl ToTaffy<LengthPercentageAuto> for DefiniteLength {
    fn to_taffy(&self, rem_size: Pixels, scale_factor: f32) -> LengthPercentageAuto {
        match self {
            DefiniteLength::Absolute(length) => {
                LengthPercentageAuto::length(absolute_px(length, rem_size, scale_factor))
            }
            DefiniteLength::Fraction(fraction) => LengthPercentageAuto::percent(*fraction),
        }
    }
}

impl ToTaffy<Dimension> for DefiniteLength {
    fn to_taffy(&self, rem_size: Pixels, scale_factor: f32) -> Dimension {
        match self {
            DefiniteLength::Absolute(AbsoluteLength::Pixels(pixels)) => {
                Dimension::length(f32::from(*pixels) * scale_factor)
            }
            // GPUI multiplies in `Pixels` here (`rems * rem_size * scale_factor`); same rounding as above.
            DefiniteLength::Absolute(AbsoluteLength::Rems(rems)) => {
                Dimension::length(f32::from(*rems * rem_size * scale_factor))
            }
            DefiniteLength::Fraction(fraction) => Dimension::percent(*fraction),
        }
    }
}

impl ToTaffy<LengthPercentage> for AbsoluteLength {
    fn to_taffy(&self, rem_size: Pixels, scale_factor: f32) -> LengthPercentage {
        LengthPercentage::length(absolute_px(self, rem_size, scale_factor))
    }
}

impl<T, U> ToTaffy<TaffySize<U>> for Size<T>
where
    T: ToTaffy<U> + Clone + Debug + Default + PartialEq,
{
    fn to_taffy(&self, rem_size: Pixels, scale_factor: f32) -> TaffySize<U> {
        TaffySize {
            width: self.width.to_taffy(rem_size, scale_factor),
            height: self.height.to_taffy(rem_size, scale_factor),
        }
    }
}

impl<T, U> ToTaffy<TaffyRect<U>> for Edges<T>
where
    T: ToTaffy<U> + Clone + Debug + Default + PartialEq,
{
    fn to_taffy(&self, rem_size: Pixels, scale_factor: f32) -> TaffyRect<U> {
        TaffyRect {
            top: self.top.to_taffy(rem_size, scale_factor),
            right: self.right.to_taffy(rem_size, scale_factor),
            bottom: self.bottom.to_taffy(rem_size, scale_factor),
            left: self.left.to_taffy(rem_size, scale_factor),
        }
    }
}

impl<T, T2> From<TaffyPoint<T>> for Point<T2>
where
    T: Into<T2>,
    T2: Clone + Debug + Default + PartialEq,
{
    fn from(point: TaffyPoint<T>) -> Point<T2> {
        Point { x: point.x.into(), y: point.y.into() }
    }
}

impl<T, T2> From<Point<T>> for TaffyPoint<T2>
where
    T: Into<T2> + Clone + Debug + Default + PartialEq,
{
    fn from(val: Point<T>) -> Self {
        TaffyPoint { x: val.x.into(), y: val.y.into() }
    }
}

impl<T, U> From<TaffySize<T>> for Size<U>
where
    T: Into<U>,
    U: Clone + Debug + Default + PartialEq,
{
    fn from(taffy_size: TaffySize<T>) -> Self {
        Size { width: taffy_size.width.into(), height: taffy_size.height.into() }
    }
}

impl<T, U> From<Size<T>> for TaffySize<U>
where
    T: Into<U> + Clone + Debug + Default + PartialEq,
{
    fn from(size: Size<T>) -> Self {
        TaffySize { width: size.width.into(), height: size.height.into() }
    }
}

/// The space available to lay an element out in (GPUI's `AvailableSpace`).
#[derive(Copy, Clone, Default, Debug, Eq, PartialEq)]
pub enum AvailableSpace {
    /// This many pixels.
    Definite(Pixels),
    /// Indefinite: lay out under a min-content constraint.
    #[default]
    MinContent,
    /// Indefinite: lay out under a max-content constraint.
    MaxContent,
}

impl AvailableSpace {
    /// Min-content on both axes.
    pub const fn min_size() -> Size<Self> {
        Size { width: Self::MinContent, height: Self::MinContent }
    }
}

impl From<AvailableSpace> for taffy::style::AvailableSpace {
    fn from(space: AvailableSpace) -> Self {
        match space {
            AvailableSpace::Definite(Pixels(value)) => Self::Definite(value),
            AvailableSpace::MinContent => Self::MinContent,
            AvailableSpace::MaxContent => Self::MaxContent,
        }
    }
}

impl From<taffy::style::AvailableSpace> for AvailableSpace {
    fn from(space: taffy::style::AvailableSpace) -> Self {
        match space {
            taffy::style::AvailableSpace::Definite(value) => Self::Definite(Pixels(value)),
            taffy::style::AvailableSpace::MinContent => Self::MinContent,
            taffy::style::AvailableSpace::MaxContent => Self::MaxContent,
        }
    }
}

impl From<Pixels> for AvailableSpace {
    fn from(pixels: Pixels) -> Self {
        AvailableSpace::Definite(pixels)
    }
}

impl From<Size<Pixels>> for Size<AvailableSpace> {
    fn from(size: Size<Pixels>) -> Self {
        Size { width: AvailableSpace::Definite(size.width), height: AvailableSpace::Definite(size.height) }
    }
}
