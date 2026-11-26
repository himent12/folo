// Adapted from GPUI (Apache-2.0): crates/gpui/src/styled.rs, crates/gpui_macros/src/styles.rs
//! `Styled`: GPUI's fluent, Tailwind-like style methods, each writing into the element's
//! [`StyleRefinement`]. `FluentBuilder` (`map`, `when`, …) is in `fluent.rs`.

mod fluent;
mod macros;

pub use fluent::*;

use macros::{
    border_methods, border_scale, fixed_setters, margin_methods, padding_methods, position_methods,
    rounded_methods, size_methods, value_setters,
};

use crate::{
    AbsoluteLength, AlignContent, AlignItems, BorderStyle, BoxShadow, CursorStyle, DefiniteLength, Display,
    Fill, FlexDirection, FlexWrap, Font, FontStyle, FontWeight, Hsla, JustifyContent, Length, Overflow,
    Position, SharedString, StrikethroughStyle, StyleRefinement, TextAlign, TextOverflow,
    TextStyleRefinement, UnderlineStyle, Visibility, WhiteSpace, auto, hsla, point, px, relative, rems,
};

const ELLIPSIS: SharedString = SharedString::new_static("…");

/// An element whose style can be set with GPUI's utility methods.
pub trait Styled: Sized {
    /// The element's style refinement, which every method writes into.
    fn style(&mut self) -> &mut StyleRefinement;

    size_methods!();
    rounded_methods!();
    margin_methods!();
    padding_methods!();
    position_methods!();
    border_methods!();

    /// Draws the element (the default).
    fn visible(mut self) -> Self {
        self.style().visibility = Some(Visibility::Visible);
        self
    }

    /// Lays the element out but does not draw it.
    fn invisible(mut self) -> Self {
        self.style().visibility = Some(Visibility::Hidden);
        self
    }

    /// Positions the element relative to where the layout puts it.
    fn relative(mut self) -> Self {
        self.style().position = Some(Position::Relative);
        self
    }

    /// Positions the element relative to its closest positioned ancestor.
    fn absolute(mut self) -> Self {
        self.style().position = Some(Position::Absolute);
        self
    }

    /// Clips overflowing content on both axes.
    fn overflow_hidden(mut self) -> Self {
        self.style().overflow.x = Some(Overflow::Hidden);
        self.style().overflow.y = Some(Overflow::Hidden);
        self
    }

    /// Clips overflowing content horizontally.
    fn overflow_x_hidden(mut self) -> Self {
        self.style().overflow.x = Some(Overflow::Hidden);
        self
    }

    /// Clips overflowing content vertically.
    fn overflow_y_hidden(mut self) -> Self {
        self.style().overflow.y = Some(Overflow::Hidden);
        self
    }

    /// Sets the cursor shown over this element.
    fn cursor(mut self, cursor: CursorStyle) -> Self {
        self.style().mouse_cursor = Some(cursor);
        self
    }

    /// The `default` cursor.
    fn cursor_default(self) -> Self {
        self.cursor(CursorStyle::Arrow)
    }

    /// The `pointer` cursor.
    fn cursor_pointer(self) -> Self {
        self.cursor(CursorStyle::PointingHand)
    }

    /// The `text` cursor.
    fn cursor_text(self) -> Self {
        self.cursor(CursorStyle::IBeam)
    }

    /// The `move` cursor.
    fn cursor_move(self) -> Self {
        self.cursor(CursorStyle::ClosedHand)
    }

    /// The `not-allowed` cursor.
    fn cursor_not_allowed(self) -> Self {
        self.cursor(CursorStyle::OperationNotAllowed)
    }

    /// The `context-menu` cursor.
    fn cursor_context_menu(self) -> Self {
        self.cursor(CursorStyle::ContextualMenu)
    }

    /// The `crosshair` cursor.
    fn cursor_crosshair(self) -> Self {
        self.cursor(CursorStyle::Crosshair)
    }

    /// The `vertical-text` cursor.
    fn cursor_vertical_text(self) -> Self {
        self.cursor(CursorStyle::IBeamCursorForVerticalLayout)
    }

    /// The `alias` cursor.
    fn cursor_alias(self) -> Self {
        self.cursor(CursorStyle::DragLink)
    }

    /// The `copy` cursor.
    fn cursor_copy(self) -> Self {
        self.cursor(CursorStyle::DragCopy)
    }

    /// The `no-drop` cursor.
    fn cursor_no_drop(self) -> Self {
        self.cursor(CursorStyle::OperationNotAllowed)
    }

    /// The `grab` cursor.
    fn cursor_grab(self) -> Self {
        self.cursor(CursorStyle::OpenHand)
    }

    /// The `grabbing` cursor.
    fn cursor_grabbing(self) -> Self {
        self.cursor(CursorStyle::ClosedHand)
    }

    /// The `ew-resize` cursor.
    fn cursor_ew_resize(self) -> Self {
        self.cursor(CursorStyle::ResizeLeftRight)
    }

    /// The `ns-resize` cursor.
    fn cursor_ns_resize(self) -> Self {
        self.cursor(CursorStyle::ResizeUpDown)
    }

    /// The `nesw-resize` cursor.
    fn cursor_nesw_resize(self) -> Self {
        self.cursor(CursorStyle::ResizeUpRightDownLeft)
    }

    /// The `nwse-resize` cursor.
    fn cursor_nwse_resize(self) -> Self {
        self.cursor(CursorStyle::ResizeUpLeftDownRight)
    }

    /// The `col-resize` cursor.
    fn cursor_col_resize(self) -> Self {
        self.cursor(CursorStyle::ResizeColumn)
    }

    /// The `row-resize` cursor.
    fn cursor_row_resize(self) -> Self {
        self.cursor(CursorStyle::ResizeRow)
    }

    /// The `n-resize` cursor.
    fn cursor_n_resize(self) -> Self {
        self.cursor(CursorStyle::ResizeUp)
    }

    /// The `e-resize` cursor.
    fn cursor_e_resize(self) -> Self {
        self.cursor(CursorStyle::ResizeRight)
    }

    /// The `s-resize` cursor.
    fn cursor_s_resize(self) -> Self {
        self.cursor(CursorStyle::ResizeDown)
    }

    /// The `w-resize` cursor.
    fn cursor_w_resize(self) -> Self {
        self.cursor(CursorStyle::ResizeLeft)
    }

    /// No cursor (GPUI's signature takes, and ignores, a cursor style).
    fn cursor_none(self, _cursor: CursorStyle) -> Self {
        self.cursor(CursorStyle::None)
    }

    /// Sets the border color.
    fn border_color<C>(mut self, border_color: C) -> Self
    where
        C: Into<Hsla>,
    {
        self.style().border_color = Some(border_color.into());
        self
    }

    /// Sets the border style to dashed.
    fn border_dashed(mut self) -> Self {
        self.style().border_style = Some(BorderStyle::Dashed);
        self
    }

    /// Sets the box shadow layers.
    fn shadow(mut self, shadows: Vec<BoxShadow>) -> Self {
        self.style().box_shadow = Some(shadows);
        self
    }

    /// Clears the box shadow.
    fn shadow_none(mut self) -> Self {
        self.style().box_shadow = Some(Vec::new());
        self
    }

    /// Tailwind's `shadow-2xs`.
    fn shadow_2xs(self) -> Self {
        self.shadow(vec![shadow_layer(0.05, 1., 0., 0.)])
    }

    /// Tailwind's `shadow-xs`.
    fn shadow_xs(self) -> Self {
        self.shadow(vec![shadow_layer(0.05, 1., 2., 0.)])
    }

    /// Tailwind's `shadow-sm`.
    fn shadow_sm(self) -> Self {
        self.shadow(vec![shadow_layer(0.1, 1., 3., 0.), shadow_layer(0.1, 1., 2., -1.)])
    }

    /// Tailwind's `shadow-md`.
    fn shadow_md(self) -> Self {
        self.shadow(vec![shadow_layer(0.1, 4., 6., -1.), shadow_layer(0.1, 2., 4., -2.)])
    }

    /// Tailwind's `shadow-lg`.
    fn shadow_lg(self) -> Self {
        self.shadow(vec![shadow_layer(0.1, 10., 15., -3.), shadow_layer(0.1, 4., 6., -4.)])
    }

    /// Tailwind's `shadow-xl`.
    fn shadow_xl(self) -> Self {
        self.shadow(vec![shadow_layer(0.1, 20., 25., -5.), shadow_layer(0.1, 8., 10., -6.)])
    }

    /// Tailwind's `shadow-2xl`.
    fn shadow_2xl(self) -> Self {
        self.shadow(vec![shadow_layer(0.25, 25., 50., -12.)])
    }

    /// Block layout.
    fn block(mut self) -> Self {
        self.style().display = Some(Display::Block);
        self
    }

    /// Flex layout.
    fn flex(mut self) -> Self {
        self.style().display = Some(Display::Flex);
        self
    }

    /// Grid layout.
    fn grid(mut self) -> Self {
        self.style().display = Some(Display::Grid);
        self
    }

    /// Not laid out or drawn (`display: none`).
    fn hidden(mut self) -> Self {
        self.style().display = Some(Display::None);
        self
    }

    /// Text wraps normally.
    fn whitespace_normal(mut self) -> Self {
        self.text_style().get_or_insert_with(Default::default).white_space = Some(WhiteSpace::Normal);
        self
    }

    /// Text does not wrap.
    fn whitespace_nowrap(mut self) -> Self {
        self.text_style().get_or_insert_with(Default::default).white_space = Some(WhiteSpace::Nowrap);
        self
    }

    /// Overflowing text is truncated with an ellipsis (…).
    fn text_ellipsis(mut self) -> Self {
        self.text_style().get_or_insert_with(Default::default).text_overflow =
            Some(TextOverflow::Truncate(ELLIPSIS));
        self
    }

    /// Sets how overflowing text is truncated.
    fn text_overflow(mut self, overflow: TextOverflow) -> Self {
        self.text_style().get_or_insert_with(Default::default).text_overflow = Some(overflow);
        self
    }

    /// Sets the text alignment.
    fn text_align(mut self, align: TextAlign) -> Self {
        self.text_style().get_or_insert_with(Default::default).text_align = Some(align);
        self
    }

    /// Left-aligned text.
    fn text_left(self) -> Self {
        self.text_align(TextAlign::Left)
    }

    /// Centered text.
    fn text_center(self) -> Self {
        self.text_align(TextAlign::Center)
    }

    /// Right-aligned text.
    fn text_right(self) -> Self {
        self.text_align(TextAlign::Right)
    }

    /// One line of text, clipped, truncated with an ellipsis.
    fn truncate(self) -> Self {
        self.overflow_hidden().whitespace_nowrap().text_ellipsis()
    }

    /// Shows at most `lines` lines of text, then truncates (and clips overflow).
    fn line_clamp(mut self, lines: usize) -> Self {
        let text_style = self.text_style().get_or_insert_with(Default::default);
        text_style.line_clamp = Some(lines);
        self.overflow_hidden()
    }

    /// Column flex direction.
    fn flex_col(mut self) -> Self {
        self.style().flex_direction = Some(FlexDirection::Column);
        self
    }

    /// Reversed column flex direction.
    fn flex_col_reverse(mut self) -> Self {
        self.style().flex_direction = Some(FlexDirection::ColumnReverse);
        self
    }

    /// Row flex direction.
    fn flex_row(mut self) -> Self {
        self.style().flex_direction = Some(FlexDirection::Row);
        self
    }

    /// Reversed row flex direction.
    fn flex_row_reverse(mut self) -> Self {
        self.style().flex_direction = Some(FlexDirection::RowReverse);
        self
    }

    /// Grows and shrinks as needed, ignoring its initial size (`flex: 1 1 0%`).
    fn flex_1(mut self) -> Self {
        self.style().flex_grow = Some(1.);
        self.style().flex_shrink = Some(1.);
        self.style().flex_basis = Some(relative(0.).into());
        self
    }

    /// Grows and shrinks from its initial size (`flex: 1 1 auto`).
    fn flex_auto(mut self) -> Self {
        self.style().flex_grow = Some(1.);
        self.style().flex_shrink = Some(1.);
        self.style().flex_basis = Some(Length::Auto);
        self
    }

    /// Shrinks but does not grow (`flex: 0 1 auto`).
    fn flex_initial(mut self) -> Self {
        self.style().flex_grow = Some(0.);
        self.style().flex_shrink = Some(1.);
        self.style().flex_basis = Some(Length::Auto);
        self
    }

    /// Neither grows nor shrinks (`flex: none`, i.e. `0 0 auto`).
    fn flex_none(mut self) -> Self {
        self.style().flex_grow = Some(0.);
        self.style().flex_shrink = Some(0.);
        self.style().flex_basis = Some(Length::Auto);
        self
    }

    /// Sets the initial main-axis size.
    fn flex_basis(mut self, basis: impl Into<Length>) -> Self {
        self.style().flex_basis = Some(basis.into());
        self
    }

    /// Grows into free space.
    fn flex_grow(mut self) -> Self {
        self.style().flex_grow = Some(1.);
        self
    }

    /// Shrinks when space is short.
    fn flex_shrink(mut self) -> Self {
        self.style().flex_shrink = Some(1.);
        self
    }

    /// Never shrinks.
    fn flex_shrink_0(mut self) -> Self {
        self.style().flex_shrink = Some(0.);
        self
    }

    /// Children wrap onto several lines.
    fn flex_wrap(mut self) -> Self {
        self.style().flex_wrap = Some(FlexWrap::Wrap);
        self
    }

    /// Children wrap in reverse.
    fn flex_wrap_reverse(mut self) -> Self {
        self.style().flex_wrap = Some(FlexWrap::WrapReverse);
        self
    }

    /// Children stay on one line.
    fn flex_nowrap(mut self) -> Self {
        self.style().flex_wrap = Some(FlexWrap::NoWrap);
        self
    }

    /// Children at the start of the cross axis.
    fn items_start(mut self) -> Self {
        self.style().align_items = Some(AlignItems::FlexStart);
        self
    }

    /// Children at the end of the cross axis.
    fn items_end(mut self) -> Self {
        self.style().align_items = Some(AlignItems::FlexEnd);
        self
    }

    /// Children centered on the cross axis.
    fn items_center(mut self) -> Self {
        self.style().align_items = Some(AlignItems::Center);
        self
    }

    /// Children aligned on their baselines.
    fn items_baseline(mut self) -> Self {
        self.style().align_items = Some(AlignItems::Baseline);
        self
    }

    /// Children at the start of the main axis.
    fn justify_start(mut self) -> Self {
        self.style().justify_content = Some(JustifyContent::Start);
        self
    }

    /// Children at the end of the main axis.
    fn justify_end(mut self) -> Self {
        self.style().justify_content = Some(JustifyContent::End);
        self
    }

    /// Children centered on the main axis.
    fn justify_center(mut self) -> Self {
        self.style().justify_content = Some(JustifyContent::Center);
        self
    }

    /// Children spread with equal space between them.
    fn justify_between(mut self) -> Self {
        self.style().justify_content = Some(JustifyContent::SpaceBetween);
        self
    }

    /// Children spread with equal space around each.
    fn justify_around(mut self) -> Self {
        self.style().justify_content = Some(JustifyContent::SpaceAround);
        self
    }

    /// Lines of content in their default position (unsets `align_content`).
    fn content_normal(mut self) -> Self {
        self.style().align_content = None;
        self
    }

    /// Lines of content centered on the cross axis.
    fn content_center(mut self) -> Self {
        self.style().align_content = Some(AlignContent::Center);
        self
    }

    /// Lines of content at the start of the cross axis.
    fn content_start(mut self) -> Self {
        self.style().align_content = Some(AlignContent::FlexStart);
        self
    }

    /// Lines of content at the end of the cross axis.
    fn content_end(mut self) -> Self {
        self.style().align_content = Some(AlignContent::FlexEnd);
        self
    }

    /// Lines of content with equal space between them.
    fn content_between(mut self) -> Self {
        self.style().align_content = Some(AlignContent::SpaceBetween);
        self
    }

    /// Lines of content with equal space around each.
    fn content_around(mut self) -> Self {
        self.style().align_content = Some(AlignContent::SpaceAround);
        self
    }

    /// Lines of content with equal space between and at the edges.
    fn content_evenly(mut self) -> Self {
        self.style().align_content = Some(AlignContent::SpaceEvenly);
        self
    }

    /// Lines of content stretched to fill the cross axis.
    fn content_stretch(mut self) -> Self {
        self.style().align_content = Some(AlignContent::Stretch);
        self
    }

    /// Sets the background fill.
    fn bg<F>(mut self, fill: F) -> Self
    where
        F: Into<Fill>,
    {
        self.style().background = Some(fill.into());
        self
    }

    /// The text style this element sets for its children (created on first write).
    fn text_style(&mut self) -> &mut Option<TextStyleRefinement> {
        &mut self.style().text
    }

    /// Sets the text color (cascades to children).
    fn text_color(mut self, color: impl Into<Hsla>) -> Self {
        self.text_style().get_or_insert_with(Default::default).color = Some(color.into());
        self
    }

    /// Sets the font weight (cascades to children).
    fn font_weight(mut self, weight: FontWeight) -> Self {
        self.text_style().get_or_insert_with(Default::default).font_weight = Some(weight);
        self
    }

    /// Sets the text background color (cascades to children).
    fn text_bg(mut self, bg: impl Into<Hsla>) -> Self {
        self.text_style().get_or_insert_with(Default::default).background_color = Some(bg.into());
        self
    }

    /// Sets the font size (cascades to children).
    fn text_size(mut self, size: impl Into<AbsoluteLength>) -> Self {
        self.text_style().get_or_insert_with(Default::default).font_size = Some(size.into());
        self
    }

    /// Font size 0.75 rem.
    fn text_xs(self) -> Self {
        self.text_size(rems(0.75))
    }

    /// Font size 0.875 rem.
    fn text_sm(self) -> Self {
        self.text_size(rems(0.875))
    }

    /// Font size 1 rem.
    fn text_base(self) -> Self {
        self.text_size(rems(1.0))
    }

    /// Font size 1.125 rem.
    fn text_lg(self) -> Self {
        self.text_size(rems(1.125))
    }

    /// Font size 1.25 rem.
    fn text_xl(self) -> Self {
        self.text_size(rems(1.25))
    }

    /// Font size 1.5 rem.
    fn text_2xl(self) -> Self {
        self.text_size(rems(1.5))
    }

    /// Font size 1.875 rem.
    fn text_3xl(self) -> Self {
        self.text_size(rems(1.875))
    }

    /// Italic text.
    fn italic(mut self) -> Self {
        self.text_style().get_or_insert_with(Default::default).font_style = Some(FontStyle::Italic);
        self
    }

    /// Upright text.
    fn not_italic(mut self) -> Self {
        self.text_style().get_or_insert_with(Default::default).font_style = Some(FontStyle::Normal);
        self
    }

    /// A 1 px underline.
    fn underline(mut self) -> Self {
        let style = self.text_style().get_or_insert_with(Default::default);
        style.underline = Some(UnderlineStyle { thickness: px(1.), ..Default::default() });
        self
    }

    /// A 1 px line through the text.
    fn line_through(mut self) -> Self {
        let style = self.text_style().get_or_insert_with(Default::default);
        style.strikethrough = Some(StrikethroughStyle { thickness: px(1.), ..Default::default() });
        self
    }

    /// Removes the underline (cascades to children).
    fn text_decoration_none(mut self) -> Self {
        self.text_style().get_or_insert_with(Default::default).underline = None;
        self
    }

    /// Sets the underline color.
    fn text_decoration_color(mut self, color: impl Into<Hsla>) -> Self {
        underline_of(&mut self).color = Some(color.into());
        self
    }

    /// A straight underline.
    fn text_decoration_solid(mut self) -> Self {
        underline_of(&mut self).wavy = false;
        self
    }

    /// A wavy underline.
    fn text_decoration_wavy(mut self) -> Self {
        underline_of(&mut self).wavy = true;
        self
    }

    /// A 0 px underline.
    fn text_decoration_0(mut self) -> Self {
        underline_of(&mut self).thickness = px(0.);
        self
    }

    /// A 1 px underline.
    fn text_decoration_1(mut self) -> Self {
        underline_of(&mut self).thickness = px(1.);
        self
    }

    /// A 2 px underline.
    fn text_decoration_2(mut self) -> Self {
        underline_of(&mut self).thickness = px(2.);
        self
    }

    /// A 4 px underline.
    fn text_decoration_4(mut self) -> Self {
        underline_of(&mut self).thickness = px(4.);
        self
    }

    /// An 8 px underline.
    fn text_decoration_8(mut self) -> Self {
        underline_of(&mut self).thickness = px(8.);
        self
    }

    /// Sets the font family (cascades to children).
    fn font_family(mut self, family_name: impl Into<SharedString>) -> Self {
        self.text_style().get_or_insert_with(Default::default).font_family = Some(family_name.into());
        self
    }

    /// Sets the whole font: family, features, weight, style and fallbacks (cascades to children).
    fn font(mut self, font: Font) -> Self {
        let Font { family, features, fallbacks, weight, style } = font;
        let text_style = self.text_style().get_or_insert_with(Default::default);
        text_style.font_family = Some(family);
        text_style.font_features = Some(features);
        text_style.font_weight = Some(weight);
        text_style.font_style = Some(style);
        text_style.font_fallbacks = fallbacks;
        self
    }

    /// Sets the line height (cascades to children).
    fn line_height(mut self, line_height: impl Into<DefiniteLength>) -> Self {
        self.text_style().get_or_insert_with(Default::default).line_height = Some(line_height.into());
        self
    }

    /// Sets the opacity of this element and its children.
    fn opacity(mut self, opacity: f32) -> Self {
        self.style().opacity = Some(opacity);
        self
    }

    /// Sets the number of grid columns.
    fn grid_cols(mut self, cols: u16) -> Self {
        self.style().grid_cols = Some(cols);
        self
    }

    /// Sets the number of grid rows.
    fn grid_rows(mut self, rows: u16) -> Self {
        self.style().grid_rows = Some(rows);
        self
    }
}

/// The element's underline style, created (default) on first write.
fn underline_of<S: Styled>(styled: &mut S) -> &mut UnderlineStyle {
    let style = styled.text_style().get_or_insert_with(Default::default);
    style.underline.get_or_insert_with(Default::default)
}

/// One black box-shadow layer of GPUI's Tailwind shadows.
fn shadow_layer(alpha: f32, y: f32, blur: f32, spread: f32) -> BoxShadow {
    BoxShadow {
        color: hsla(0., 0., 0., alpha),
        offset: point(px(0.), px(y)),
        blur_radius: px(blur),
        spread_radius: px(spread),
    }
}

impl Styled for StyleRefinement {
    fn style(&mut self) -> &mut StyleRefinement {
        self
    }
}
