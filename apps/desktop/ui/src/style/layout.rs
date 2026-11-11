// Adapted from GPUI (Apache-2.0): crates/gpui/src/style.rs
//! The layout enums (copies of taffy's, as GPUI keeps them) and their conversions to taffy.

/// How children are aligned on the cross axis (flex) or block axis (grid).
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum AlignItems {
    /// Packed toward the start of the axis.
    Start,
    /// Packed toward the end of the axis.
    End,
    /// Packed toward the flex-relative start of the axis.
    FlexStart,
    /// Packed toward the flex-relative end of the axis.
    FlexEnd,
    /// Centered on the cross axis.
    Center,
    /// Aligned on their baselines.
    Baseline,
    /// Stretched to fill the container.
    Stretch,
}

/// How a grid item is aligned on the inline axis.
pub type JustifyItems = AlignItems;
/// How this element is aligned on its parent's cross axis (overrides the parent's `AlignItems`).
pub type AlignSelf = AlignItems;
/// How this grid item is aligned on the inline axis.
pub type JustifySelf = AlignItems;

/// How space is distributed between and around lines of content (cross axis in flex).
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum AlignContent {
    /// Packed toward the start of the axis.
    Start,
    /// Packed toward the end of the axis.
    End,
    /// Packed toward the flex-relative start of the axis.
    FlexStart,
    /// Packed toward the flex-relative end of the axis.
    FlexEnd,
    /// Centered.
    Center,
    /// Stretched to fill the container.
    Stretch,
    /// Flush with the edges, the free space shared between items.
    SpaceBetween,
    /// Equal space between items and at the edges.
    SpaceEvenly,
    /// Half the space between items at the edges.
    SpaceAround,
}

/// How space is distributed between and around items on the main axis (flex).
pub type JustifyContent = AlignContent;

/// The layout algorithm used for an element's children.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Default)]
pub enum Display {
    /// Block layout.
    Block,
    /// Flexbox layout.
    #[default]
    Flex,
    /// CSS grid layout.
    Grid,
    /// Not laid out (and not drawn).
    None,
}

/// Whether flex items stay on one line or wrap.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Default)]
pub enum FlexWrap {
    /// One line.
    #[default]
    NoWrap,
    /// Wrap in the flex direction.
    Wrap,
    /// Wrap in the opposite direction.
    WrapReverse,
}

/// The direction of the main axis.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Default)]
pub enum FlexDirection {
    /// Left to right.
    #[default]
    Row,
    /// Top to bottom.
    Column,
    /// Right to left.
    RowReverse,
    /// Bottom to top.
    ColumnReverse,
}

/// How children that overflow their container affect layout (and, for painting, whether they are clipped).
#[derive(Copy, Clone, PartialEq, Eq, Debug, Default)]
pub enum Overflow {
    /// Content-based automatic minimum size; overflow contributes to the parent's scroll region.
    #[default]
    Visible,
    /// Content-based automatic minimum size; overflow does not contribute to the parent's scroll region.
    Clip,
    /// Automatic minimum size of zero; overflow does not contribute to the parent's scroll region.
    Hidden,
    /// Like `Hidden`, with space reserved for a scrollbar (`scrollbar_width`).
    Scroll,
}

/// What an element's `inset` is relative to (CSS `position`; `Relative` is the default, unlike CSS).
#[derive(Copy, Clone, PartialEq, Eq, Debug, Default)]
pub enum Position {
    /// Relative to where the layout puts the element; does not move other elements.
    #[default]
    Relative,
    /// Relative to the closest positioned ancestor; takes no space in the layout.
    Absolute,
}

impl From<AlignItems> for taffy::style::AlignItems {
    fn from(value: AlignItems) -> Self {
        match value {
            AlignItems::Start => Self::Start,
            AlignItems::End => Self::End,
            AlignItems::FlexStart => Self::FlexStart,
            AlignItems::FlexEnd => Self::FlexEnd,
            AlignItems::Center => Self::Center,
            AlignItems::Baseline => Self::Baseline,
            AlignItems::Stretch => Self::Stretch,
        }
    }
}

impl From<AlignContent> for taffy::style::AlignContent {
    fn from(value: AlignContent) -> Self {
        match value {
            AlignContent::Start => Self::Start,
            AlignContent::End => Self::End,
            AlignContent::FlexStart => Self::FlexStart,
            AlignContent::FlexEnd => Self::FlexEnd,
            AlignContent::Center => Self::Center,
            AlignContent::Stretch => Self::Stretch,
            AlignContent::SpaceBetween => Self::SpaceBetween,
            AlignContent::SpaceEvenly => Self::SpaceEvenly,
            AlignContent::SpaceAround => Self::SpaceAround,
        }
    }
}

impl From<Display> for taffy::style::Display {
    fn from(value: Display) -> Self {
        match value {
            Display::Block => Self::Block,
            Display::Flex => Self::Flex,
            Display::Grid => Self::Grid,
            Display::None => Self::None,
        }
    }
}

impl From<FlexWrap> for taffy::style::FlexWrap {
    fn from(value: FlexWrap) -> Self {
        match value {
            FlexWrap::NoWrap => Self::NoWrap,
            FlexWrap::Wrap => Self::Wrap,
            FlexWrap::WrapReverse => Self::WrapReverse,
        }
    }
}

impl From<FlexDirection> for taffy::style::FlexDirection {
    fn from(value: FlexDirection) -> Self {
        match value {
            FlexDirection::Row => Self::Row,
            FlexDirection::Column => Self::Column,
            FlexDirection::RowReverse => Self::RowReverse,
            FlexDirection::ColumnReverse => Self::ColumnReverse,
        }
    }
}

impl From<Overflow> for taffy::style::Overflow {
    fn from(value: Overflow) -> Self {
        match value {
            Overflow::Visible => Self::Visible,
            Overflow::Clip => Self::Clip,
            Overflow::Hidden => Self::Hidden,
            Overflow::Scroll => Self::Scroll,
        }
    }
}

impl From<Position> for taffy::style::Position {
    fn from(value: Position) -> Self {
        match value {
            Position::Relative => Self::Relative,
            Position::Absolute => Self::Absolute,
        }
    }
}
