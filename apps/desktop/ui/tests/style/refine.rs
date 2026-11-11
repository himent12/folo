use folo_ui::{
    Display, Fill, Length, Overflow, Refineable, Style, StyleRefinement, Styled, TextStyle,
    TextStyleRefinement, blue, green, px, red, rems,
};

#[test]
fn later_refinements_win_and_unset_fields_keep_earlier_values() {
    let base = StyleRefinement::default().w(px(10.)).bg(red()).p(px(4.)).overflow_x_hidden();
    let hover = StyleRefinement::default().bg(blue()).pl(px(9.)).overflow_y_hidden();

    let mut merged = base.clone();
    merged.refine(&hover);
    assert_eq!(merged.background, Some(Fill::from(blue())));
    assert_eq!(merged.size.width, Some(px(10.).into()));
    // Nested fields merge component by component.
    assert_eq!(merged.padding.left, Some(px(9.).into()));
    assert_eq!(merged.padding.top, Some(px(4.).into()));
    assert_eq!((merged.overflow.x, merged.overflow.y), (Some(Overflow::Hidden), Some(Overflow::Hidden)));

    let style = Style::default().refined(base).refined(hover);
    assert_eq!(style.background, Some(Fill::from(blue())));
    assert_eq!(style.size.width, Length::from(px(10.)));
    assert_eq!(style.size.height, Length::Auto);
    assert_eq!(style.padding.left, px(9.).into());
    assert_eq!(style.padding.right, px(4.).into());
    // A default `Style` is block layout until `flex()` (GPUI's quirk), and refining an empty refinement
    // changes nothing.
    assert_eq!(style.display, Display::Block);
    assert_eq!(Style::default().refined(StyleRefinement::default().flex()).display, Display::Flex);
}

#[test]
fn a_text_refinement_replaces_the_whole_text_style_as_gpui_does() {
    // `Style::text` is not a nested refineable field in GPUI: a hover's text refinement replaces it.
    let base = StyleRefinement::default().text_color(red()).text_size(px(12.));
    let hover = StyleRefinement::default().text_color(blue());
    let style = Style::default().refined(base.clone()).refined(hover.clone());
    assert_eq!(style.text.color, Some(blue()));
    assert_eq!(style.text.font_size, None);

    let mut merged = base;
    merged.refine(&hover);
    assert_eq!(merged.text.as_ref().and_then(|t| t.font_size), None);
    assert!(StyleRefinement::default().is_empty() && !hover.is_empty());
}

#[test]
fn text_styles_cascade_field_by_field() {
    let mut text = TextStyle::default();
    assert_eq!(text.font_size, rems(1.).into());
    text.refine(&TextStyleRefinement {
        color: Some(green()),
        font_size: Some(px(13.).into()),
        ..Default::default()
    });
    text.refine(&TextStyleRefinement { line_height: Some(px(20.).into()), ..Default::default() });
    assert_eq!((text.color, text.font_size, text.line_height), (green(), px(13.).into(), px(20.).into()));
    // GPUI rounds the line height to whole pixels: 1.618 (phi) * 13 = 21.03 -> 21.
    assert_eq!(
        TextStyle { font_size: px(13.).into(), ..Default::default() }.line_height_in_pixels(px(16.)),
        px(21.)
    );
    let run = text.to_run(5);
    assert_eq!((run.len, run.color, run.font.family.as_str()), (5, green(), ".SystemUIFont"));
}
