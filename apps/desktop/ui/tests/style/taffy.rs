//! The style-to-taffy mapping (`style/to_taffy.rs`): GPUI's defaults, lengths and rems, fields set directly, and
//! available space both ways.

use folo_ui::{
    AlignItems, Display, Overflow, Position, Style, StyleRefinement, Styled, auto, px, relative, rems,
};
use taffy::style::{Dimension, LengthPercentage, LengthPercentageAuto};

fn style(f: impl FnOnce(StyleRefinement) -> StyleRefinement) -> Style {
    use folo_ui::Refineable;
    Style::default().refined(f(StyleRefinement::default()))
}

#[test]
fn default_style_maps_to_gpuis_taffy_defaults() {
    let t = Style::default().to_taffy(px(16.), 1.);
    assert_eq!(t.display, taffy::Display::Block);
    assert_eq!(t.position, taffy::Position::Relative);
    assert_eq!(t.size.width, Dimension::auto());
    assert_eq!(t.inset.left, LengthPercentageAuto::auto());
    assert_eq!(t.margin.top, LengthPercentageAuto::length(0.));
    assert_eq!(t.padding.top, LengthPercentage::length(0.));
    assert_eq!(t.gap.width, LengthPercentage::length(0.));
    assert_eq!((t.flex_grow, t.flex_shrink), (0., 1.));
    assert_eq!(t.flex_basis, Dimension::auto());
    assert!(t.grid_template_columns.is_empty() && t.align_items.is_none());
}

#[test]
fn lengths_scale_to_device_pixels_and_rems_resolve() {
    // rem 16 px, scale 1.5.
    let s = style(|s| {
        s.flex()
            .flex_col()
            .w(px(100.))
            .h(relative(0.5))
            .min_w_0()
            .max_w(rems(10.))
            .p(rems(1.))
            .pt(px(3.))
            .mx_auto()
            .mt(px(-4.))
            .gap_x(px(6.))
            .gap_y(relative(0.25))
            .border_1()
            .border_l(rems(0.25))
            .absolute()
            .top(px(2.))
            .left(relative(0.1))
            .flex_1()
            .items_center()
            .justify_between()
            .overflow_hidden()
            .grid_cols(4)
    });
    let t = s.to_taffy(px(16.), 1.5);
    assert_eq!(t.display, taffy::Display::Flex);
    assert_eq!(t.flex_direction, taffy::FlexDirection::Column);
    assert_eq!(t.size.width, Dimension::length(150.));
    assert_eq!(t.size.height, Dimension::percent(0.5));
    assert_eq!(t.min_size.width, Dimension::length(0.));
    assert_eq!(t.max_size.width, Dimension::length(240.));
    assert_eq!(t.padding.left, LengthPercentage::length(24.));
    assert_eq!(t.padding.top, LengthPercentage::length(4.5));
    assert_eq!(t.margin.left, LengthPercentageAuto::auto());
    assert_eq!(t.margin.top, LengthPercentageAuto::length(-6.));
    assert_eq!(t.margin.bottom, LengthPercentageAuto::length(0.));
    assert_eq!(t.gap.width, LengthPercentage::length(9.));
    assert_eq!(t.gap.height, LengthPercentage::percent(0.25));
    assert_eq!(t.border.top, LengthPercentage::length(1.5));
    assert_eq!(t.border.left, LengthPercentage::length(6.));
    assert_eq!(t.position, taffy::Position::Absolute);
    assert_eq!(t.inset.top, LengthPercentageAuto::length(3.));
    assert_eq!(t.inset.left, LengthPercentageAuto::percent(0.1));
    assert_eq!(t.inset.right, LengthPercentageAuto::auto());
    assert_eq!((t.flex_grow, t.flex_shrink, t.flex_basis), (1., 1., Dimension::percent(0.)));
    assert_eq!(t.align_items, Some(taffy::AlignItems::Center));
    assert_eq!(t.justify_content, Some(taffy::JustifyContent::SpaceBetween));
    assert_eq!(t.overflow.x, taffy::Overflow::Hidden);
    assert_eq!(t.overflow.y, taffy::Overflow::Hidden);
    assert_eq!(t.grid_template_columns.len(), 1);
    assert!(t.grid_template_rows.is_empty());
}

#[test]
fn fields_set_directly_map_through() {
    let mut s = Style {
        display: Display::Grid,
        position: Position::Absolute,
        aspect_ratio: Some(1.0),
        align_self: Some(AlignItems::End),
        scrollbar_width: px(8.).into(),
        flex_basis: rems(2.).into(),
        ..Default::default()
    };
    s.overflow.y = Overflow::Scroll;
    s.size.width = auto();
    let t = s.to_taffy(px(10.), 2.);
    assert_eq!(t.display, taffy::Display::Grid);
    assert_eq!(t.aspect_ratio, Some(1.0));
    assert_eq!(t.align_self, Some(taffy::AlignSelf::End));
    assert_eq!((t.overflow.x, t.overflow.y), (taffy::Overflow::Visible, taffy::Overflow::Scroll));
    assert_eq!(t.scrollbar_width, 16.);
    assert_eq!(t.size.width, Dimension::auto());
    assert_eq!(t.flex_basis, Dimension::length(40.));
}

#[test]
fn available_space_converts_both_ways() {
    use folo_ui::{AvailableSpace, Size, size};
    let s: Size<AvailableSpace> = size(px(10.), px(20.)).into();
    assert_eq!(s.width, AvailableSpace::Definite(px(10.)));
    assert_eq!(taffy::AvailableSpace::from(s.height), taffy::AvailableSpace::Definite(20.));
    assert_eq!(AvailableSpace::from(taffy::AvailableSpace::MaxContent), AvailableSpace::MaxContent);
    assert_eq!(AvailableSpace::min_size().height, AvailableSpace::MinContent);
}
