//! Every `Styled` and `FluentBuilder` method `apps/desktop/src` calls, with the argument forms it uses, on a
//! dummy element: this file failing to compile means the app would not either. A few results are checked.

use folo_ui::prelude::*;
use folo_ui::{
    AlignItems, BoxShadow, CursorStyle, Display, Fill, FontFeatures, FontWeight, Length, Overflow, Pixels,
    Position, SharedString, StyleRefinement, TextAlign, TextOverflow, Visibility, WhiteSpace, font, hsla,
    linear_color_stop, linear_gradient, point, px, relative, rgb,
};
use std::sync::Arc;

#[derive(Default)]
struct El(StyleRefinement);

impl Styled for El {
    fn style(&mut self) -> &mut StyleRefinement {
        &mut self.0
    }
}

impl FluentBuilder for El {}

fn el() -> El {
    El::default()
}

/// The app's own extension (`src/ui/text.rs`): `text_style()` written through `get_or_insert_with`.
fn tnum<S: Styled>(mut s: S) -> S {
    s.text_style().get_or_insert_with(Default::default).font_features =
        Some(FontFeatures(Arc::new(vec![("tnum".into(), 1)])));
    s
}

#[test]
fn every_method_the_app_calls_exists_with_its_argument_forms() {
    let color = hsla(0.5, 0.5, 0.5, 1.);
    let w: Pixels = px(30.);
    let shadows =
        vec![BoxShadow { color, offset: point(px(0.), px(1.)), blur_radius: px(2.), spread_radius: px(0.) }];
    let family: SharedString = "Geist".into();
    let maybe: Option<f32> = Some(2.);

    let e = el()
        // Layout and flex.
        .flex()
        .flex_col()
        .flex_1()
        .flex_none()
        .flex_grow()
        .flex_shrink()
        .flex_wrap()
        .flex_basis(relative(0.21))
        .flex_basis(px(0.))
        .grid()
        .grid_cols(4)
        .items_center()
        .items_start()
        .items_end()
        .items_baseline()
        .justify_between()
        .justify_center()
        .justify_end()
        .gap(px(8.))
        .gap_x(px(12.))
        .gap_y(px(4.))
        // Sizes.
        .w(px(100.))
        .w(relative(0.5))
        .w(w + px(4.))
        .w_full()
        .h(px(20.))
        .h_full()
        .size(px(14.))
        .size_full()
        .min_w(px(20.))
        .min_w_0()
        .min_h(px(28.))
        .min_h_0()
        .max_w(px(400.))
        .max_w_full()
        .max_h(px(420.))
        .max_h(relative(0.92))
        // Spacing.
        .p(px(4.))
        .px(px(12.))
        .py(px(6.))
        .pt(px(2.))
        .pb(px(2.))
        .pl(px(2.))
        .pr(px(2.))
        .m(px(-4.))
        .mx(px(-16.))
        .my(px(-4.))
        .mt(px(1.))
        .mb(px(1.))
        .ml(px(1.))
        .mr(px(1.))
        .mx_auto()
        // Position.
        .relative()
        .absolute()
        .inset_0()
        .top(px(-4.))
        .top_0()
        .left(relative(0.3))
        .left(px(7.))
        .left_0()
        .right(px(8.))
        .right_0()
        .bottom(px(24.))
        .bottom_0()
        // Borders and corners.
        .border_1()
        .border_2()
        .border_t_1()
        .border_b_0()
        .border_b_1()
        .border_b_2()
        .border_l_1()
        .border_r_1()
        .border_color(color)
        .border_dashed()
        .rounded(px(6.))
        .rounded_t(px(2.))
        .rounded_b(px(3.))
        .rounded_full()
        .rounded_l_full()
        // Paint.
        .bg(color)
        .bg(rgb(0x112233))
        .bg(linear_gradient(180., linear_color_stop(color, 0.), linear_color_stop(color, 1.)))
        .shadow(shadows.clone())
        .shadow_sm()
        .shadow_md()
        .shadow_lg()
        .opacity(0.8)
        .overflow_hidden()
        .visible()
        .invisible()
        .cursor(CursorStyle::IBeam)
        .cursor_default()
        .cursor_pointer()
        .cursor_not_allowed()
        // Text.
        .text_color(color)
        .text_size(px(13.))
        .line_height(px(17.))
        .font_family("Geist Mono")
        .font_family(family.clone())
        .font_weight(FontWeight::MEDIUM)
        .font(font("Geist"))
        .text_left()
        .text_center()
        .text_right()
        .text_ellipsis()
        .truncate()
        .whitespace_normal()
        .whitespace_nowrap()
        .line_clamp(1)
        // FluentBuilder.
        .when(true, |e| e.p(px(1.)))
        .when_some(maybe, |e, v| e.gap(px(v)))
        .when_none(&None::<u8>, |e| e)
        .when_else(false, |e| e, |e| e)
        .map(|e| e.opacity(0.5));
    let El(style) = tnum(e);

    assert_eq!(style.display, Some(Display::Grid));
    assert_eq!(style.size.width, Some(Length::from(relative(1.))));
    assert_eq!(style.size.height, Some(Length::from(relative(1.))));
    assert_eq!(style.padding.top, Some(px(1.).into()));
    assert_eq!(style.gap.width, Some(px(2.).into()));
    assert_eq!(style.margin.left, Some(Length::Auto));
    assert_eq!(style.position, Some(Position::Absolute));
    assert_eq!(style.align_items, Some(AlignItems::Baseline));
    assert_eq!((style.overflow.x, style.overflow.y), (Some(Overflow::Hidden), Some(Overflow::Hidden)));
    assert_eq!(style.visibility, Some(Visibility::Hidden));
    assert_eq!(style.mouse_cursor, Some(CursorStyle::OperationNotAllowed));
    assert_eq!(style.opacity, Some(0.5));
    assert_eq!(style.corner_radii.top_left, Some(px(9999.).into()));
    assert_eq!(style.corner_radii.top_right, Some(px(9999.).into()));
    assert_eq!(style.border_widths.left, Some(px(1.).into()));
    assert_eq!(style.border_widths.bottom, Some(px(2.).into()));
    assert!(matches!(style.background, Some(Fill::Color(_))));
    assert_eq!(style.box_shadow.as_ref().map(Vec::len), Some(2), "shadow_lg's two layers, last set");
    let text = style.text.expect("text style set");
    assert_eq!(text.white_space, Some(WhiteSpace::Nowrap));
    assert_eq!(text.text_overflow, Some(TextOverflow::Truncate("…".into())));
    assert_eq!(text.text_align, Some(TextAlign::Right));
    assert_eq!(text.line_clamp, Some(1));
    assert_eq!(text.font_family, Some("Geist".into()));
    assert_eq!(text.font_weight, Some(FontWeight::default()), "font() sets the whole font");
    assert_eq!(text.font_features.map(|f| f.tag_value_list().to_vec()), Some(vec![("tnum".to_string(), 1)]));
}

#[test]
fn gpuis_tailwind_values() {
    let El(s) = el().shadow_md().rounded_lg().border_x_3().mx_0().text_sm();
    let md = s.box_shadow.unwrap();
    assert_eq!(md[0].offset, point(px(0.), px(4.)));
    assert_eq!((md[0].blur_radius, md[0].spread_radius), (px(6.), px(-1.)));
    assert_eq!((md[1].blur_radius, md[1].spread_radius), (px(4.), px(-2.)));
    assert_eq!(md[1].color, hsla(0., 0., 0., 0.1));
    assert_eq!(s.corner_radii.bottom_left, Some(folo_ui::rems(0.5).into()));
    assert_eq!((s.border_widths.left, s.border_widths.top), (Some(px(3.).into()), None));
    assert_eq!(s.margin.right, Some(px(0.).into()));
    assert_eq!(s.text.unwrap().font_size, Some(folo_ui::rems(0.875).into()));
}
