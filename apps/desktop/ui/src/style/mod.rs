// Adapted from GPUI (Apache-2.0): crates/gpui/src/style.rs
//! `Style`, `StyleRefinement` and the style value types (GPUI's `style.rs`): what an element's layout
//! and paint read. The taffy mapping is in `taffy.rs`, painting in `paint.rs`.

mod layout;
mod paint;
mod refine;
mod text;
mod to_taffy;

pub use layout::*;
pub use paint::*;
pub use refine::*;
pub use text::*;

use refine::{full_ty, refine_full, refine_part, refineable, refinement_ty};

use crate::{
    AbsoluteLength, BorderStyle, Bounds, Corners, CursorStyle, DefiniteLength, DevicePixels, Edges, Fill,
    Hsla, Length, Pixels, Point, Size, point, size,
};

/// How to fit an image into the bounds of its element.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum ObjectFit {
    /// Stretched to fill the bounds.
    #[default]
    Fill,
    /// Scaled to fit within the bounds.
    Contain,
    /// Scaled to cover the bounds.
    Cover,
    /// Scaled down to fit within the bounds (never up).
    ScaleDown,
    /// At its own size.
    None,
}

impl ObjectFit {
    /// Where an image of `image_size` lands inside `bounds`.
    pub fn get_bounds(&self, bounds: Bounds<Pixels>, image_size: Size<DevicePixels>) -> Bounds<Pixels> {
        let image_size = image_size.map(|dimension| Pixels::from(u32::from(dimension)));
        let image_ratio = image_size.width / image_size.height;
        let bounds_ratio = bounds.size.width / bounds.size.height;
        let centered = |new_size: Size<Pixels>| Bounds {
            origin: point(
                bounds.origin.x + (bounds.size.width - new_size.width) / 2.0,
                bounds.origin.y + (bounds.size.height - new_size.height) / 2.0,
            ),
            size: new_size,
        };
        let contain = || {
            if bounds_ratio > image_ratio {
                size(image_size.width * (bounds.size.height / image_size.height), bounds.size.height)
            } else {
                size(bounds.size.width, image_size.height * (bounds.size.width / image_size.width))
            }
        };

        match self {
            ObjectFit::Fill => bounds,
            ObjectFit::Contain => centered(contain()),
            ObjectFit::ScaleDown => {
                if image_size.width > bounds.size.width || image_size.height > bounds.size.height {
                    centered(contain())
                } else {
                    centered(image_size)
                }
            }
            ObjectFit::Cover => {
                let new_size = if bounds_ratio > image_ratio {
                    size(bounds.size.width, image_size.height * (bounds.size.width / image_size.width))
                } else {
                    size(image_size.width * (bounds.size.height / image_size.height), bounds.size.height)
                };
                centered(new_size)
            }
            ObjectFit::None => Bounds { origin: bounds.origin, size: image_size },
        }
    }
}

/// Whether an element is drawn (it takes up space either way), like CSS `visibility`.
#[derive(Default, Clone, Copy, Debug, Eq, PartialEq)]
pub enum Visibility {
    /// Drawn as normal.
    #[default]
    Visible,
    /// Not drawn, but still laid out.
    Hidden,
}

/// One layer of a box shadow.
#[derive(Clone, Debug, PartialEq)]
pub struct BoxShadow {
    /// The shadow's color.
    pub color: Hsla,
    /// How far the shadow is offset from its element.
    pub offset: Point<Pixels>,
    /// How much the shadow is blurred.
    pub blur_radius: Pixels,
    /// How far the shadow spreads past (or, negative, inside) its element.
    pub spread_radius: Pixels,
}

refineable! {
    /// The CSS-like style of an element, set through the [`Styled`](crate::Styled) methods.
    #[derive(Clone, Debug)]
    pub struct Style / #[derive(Clone, Debug, Default, PartialEq)] StyleRefinement {
        /// The layout algorithm for the children.
        plain display: Display,
        /// Whether the element is painted.
        plain visibility: Visibility,
        /// How overflowing children affect layout (and whether they are clipped).
        nested overflow: Point<Overflow> => PointRefinement<Overflow>,
        /// Space reserved for the scrollbars of `Overflow::Scroll` nodes.
        plain scrollbar_width: AbsoluteLength,
        /// Whether both axes may scroll at the same time.
        plain allow_concurrent_scroll: bool,
        /// Whether scrolling is restricted to the axis the wheel indicates (shift for x).
        plain restrict_scroll_to_axis: bool,
        /// What `inset` is relative to.
        plain position: Position,
        /// Offsets from the position the layout gives.
        nested inset: Edges<Length> => EdgesRefinement<Length>,
        /// The initial size.
        nested size: Size<Length> => SizeRefinement<Length>,
        /// The minimum size.
        nested min_size: Size<Length> => SizeRefinement<Length>,
        /// The maximum size.
        nested max_size: Size<Length> => SizeRefinement<Length>,
        /// The preferred aspect ratio (width divided by height).
        opt aspect_ratio: f32,
        /// The margin on each side.
        nested margin: Edges<Length> => EdgesRefinement<Length>,
        /// The padding on each side.
        nested padding: Edges<DefiniteLength> => EdgesRefinement<DefiniteLength>,
        /// The border width on each side.
        nested border_widths: Edges<AbsoluteLength> => EdgesRefinement<AbsoluteLength>,
        /// How the children are aligned on the cross axis.
        opt align_items: AlignItems,
        /// How this element is aligned on its parent's cross axis (falls back to the parent's `align_items`).
        opt align_self: AlignSelf,
        /// How lines of content are aligned on the cross axis.
        opt align_content: AlignContent,
        /// How the children are aligned on the main axis.
        opt justify_content: JustifyContent,
        /// The gaps between children.
        nested gap: Size<DefiniteLength> => SizeRefinement<DefiniteLength>,
        /// The main axis direction.
        plain flex_direction: FlexDirection,
        /// Whether children wrap onto several lines.
        plain flex_wrap: FlexWrap,
        /// The initial main-axis size.
        plain flex_basis: Length,
        /// How much this element grows into free space (0 by default).
        plain flex_grow: f32,
        /// How much this element shrinks when space is short (1 by default).
        plain flex_shrink: f32,
        /// The fill.
        opt background: Fill,
        /// The border color.
        opt border_color: Hsla,
        /// Solid or dashed borders.
        plain border_style: BorderStyle,
        /// The corner radii.
        nested corner_radii: Corners<AbsoluteLength> => CornersRefinement<AbsoluteLength>,
        /// The box shadow layers.
        plain box_shadow: Vec<BoxShadow>,
        /// The text style this element sets for its children (replaced as a whole by a refinement).
        plain text: TextStyleRefinement,
        /// The mouse cursor over this element.
        opt mouse_cursor: CursorStyle,
        /// The opacity of this element and its children.
        opt opacity: f32,
        /// The number of grid columns (Tailwind `grid-cols-<n>`).
        opt grid_cols: u16,
        /// The number of grid rows (Tailwind `grid-rows-<n>`).
        opt grid_rows: u16,
    }
}

impl Default for Style {
    fn default() -> Self {
        Style {
            display: Display::Block,
            visibility: Visibility::Visible,
            overflow: Point { x: Overflow::Visible, y: Overflow::Visible },
            allow_concurrent_scroll: false,
            restrict_scroll_to_axis: false,
            scrollbar_width: AbsoluteLength::default(),
            position: Position::Relative,
            inset: Edges::auto(),
            margin: Edges::<Length>::zero(),
            padding: Edges::<DefiniteLength>::zero(),
            border_widths: Edges::<AbsoluteLength>::zero(),
            size: Size::auto(),
            min_size: Size::auto(),
            max_size: Size::auto(),
            aspect_ratio: None,
            gap: Size::default(),
            align_items: None,
            align_self: None,
            align_content: None,
            justify_content: None,
            flex_direction: FlexDirection::Row,
            flex_wrap: FlexWrap::NoWrap,
            flex_grow: 0.0,
            flex_shrink: 1.0,
            flex_basis: Length::Auto,
            background: None,
            border_color: None,
            border_style: BorderStyle::default(),
            corner_radii: Corners::default(),
            box_shadow: Vec::new(),
            text: TextStyleRefinement::default(),
            mouse_cursor: None,
            opacity: None,
            grid_rows: None,
            grid_cols: None,
        }
    }
}

impl Style {
    /// Whether the background is set and not transparent.
    pub fn has_opaque_background(&self) -> bool {
        self.background.as_ref().is_some_and(|fill| fill.color().is_some_and(|color| !color.is_transparent()))
    }

    /// The text style this element sets, if it sets any field.
    pub fn text_style(&self) -> Option<&TextStyleRefinement> {
        if self.text.is_some() { Some(&self.text) } else { None }
    }

    /// Whether the border is drawn: a visible color and a non-zero width on some side.
    pub fn is_border_visible(&self) -> bool {
        self.border_color.is_some_and(|color| !color.is_transparent())
            && self.border_widths.any(|length| !length.is_zero())
    }
}
