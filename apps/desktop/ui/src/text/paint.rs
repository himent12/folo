// Adapted from GPUI (Apache-2.0): crates/gpui/src/text_system/line.rs (paint_line, paint_line_background)
//! Painting shaped lines through `GlyphPainter` (the window implements it): baseline, alignment, wrap
//! boundaries, decoration runs and per-run colours, exactly as GPUI's `paint_line`.

use crate::{
    Bounds, DecorationRun, FontId, GlyphId, Half, Hsla, LineLayout, Pixels, Point, StrikethroughStyle,
    TextAlign, TextSystem, UnderlineStyle, WrapBoundary, black, point, px, size,
};
use anyhow::Result;

/// Where painted text goes: the window implements it (`text/` does not depend on `window/`). Origins
/// and sizes are logical pixels; the implementor scales, rounds and applies the element opacity and the
/// content mask, as GPUI's `Window::paint_*` do.
pub trait GlyphPainter {
    /// Paints a monochrome glyph whose baseline origin is `origin` (GPUI's `Window::paint_glyph`).
    fn paint_glyph(
        &mut self,
        origin: Point<Pixels>,
        font_id: FontId,
        glyph_id: GlyphId,
        font_size: Pixels,
        color: Hsla,
    ) -> Result<()>;

    /// Paints a colour emoji whose baseline origin is `origin` (GPUI's `Window::paint_emoji`).
    fn paint_emoji(
        &mut self,
        origin: Point<Pixels>,
        font_id: FontId,
        glyph_id: GlyphId,
        font_size: Pixels,
    ) -> Result<()>;

    /// Paints an underline starting at `origin` (GPUI's `Window::paint_underline`).
    fn paint_underline(&mut self, origin: Point<Pixels>, width: Pixels, style: &UnderlineStyle);

    /// Paints a strikethrough starting at `origin` (GPUI's `Window::paint_strikethrough`).
    fn paint_strikethrough(&mut self, origin: Point<Pixels>, width: Pixels, style: &StrikethroughStyle);

    /// Fills `bounds` with `color` (GPUI's `window.paint_quad(fill(bounds, color))`).
    fn paint_background(&mut self, bounds: Bounds<Pixels>, color: Hsla);
}

/// Paints the glyphs and decorations of `layout` with its top-left at `origin`: one visual line per wrap
/// boundary (plus one), each aligned within `align_width` (the line's width when `None`). Glyphs whose
/// maximal bounds miss `content_mask` are skipped (GPUI reads the window's content mask there).
///
/// GPUI wraps this in `window.paint_layer(line_bounds, …)`; folo-ui's scene has no layers, so nothing
/// replaces it.
#[allow(clippy::too_many_arguments)]
pub fn paint_line(
    origin: Point<Pixels>,
    layout: &LineLayout,
    line_height: Pixels,
    align: TextAlign,
    align_width: Option<Pixels>,
    decoration_runs: &[DecorationRun],
    wrap_boundaries: &[WrapBoundary],
    text_system: &TextSystem,
    content_mask: Bounds<Pixels>,
    painter: &mut dyn GlyphPainter,
) -> Result<()> {
    let padding_top = (line_height - layout.ascent - layout.descent) / 2.;
    let baseline_offset = point(px(0.), padding_top + layout.ascent);
    let mut decoration_runs = decoration_runs.iter();
    let mut wraps = wrap_boundaries.iter().peekable();
    let mut run_end = 0;
    let mut color = black();
    let mut current_underline: Option<(Point<Pixels>, UnderlineStyle)> = None;
    let mut current_strikethrough: Option<(Point<Pixels>, StrikethroughStyle)> = None;
    let mut glyph_origin = point(
        aligned_origin_x(origin, align_width.unwrap_or(layout.width), px(0.0), &align, layout, wraps.peek()),
        origin.y,
    );
    let mut prev_glyph_position = Point::default();
    let mut max_glyph_size = size(px(0.), px(0.));
    let mut first_glyph_x = origin.x;
    for (run_ix, run) in layout.runs.iter().enumerate() {
        max_glyph_size = text_system.bounding_box(run.font_id, layout.font_size).size;

        for (glyph_ix, glyph) in run.glyphs.iter().enumerate() {
            glyph_origin.x += glyph.position.x - prev_glyph_position.x;
            if glyph_ix == 0 && run_ix == 0 {
                first_glyph_x = glyph_origin.x;
            }

            if wraps.peek() == Some(&&WrapBoundary { run_ix, glyph_ix }) {
                wraps.next();
                if let Some((underline_origin, underline_style)) = current_underline.as_mut() {
                    if glyph_origin.x == underline_origin.x {
                        underline_origin.x -= max_glyph_size.width.half();
                    };
                    painter.paint_underline(
                        *underline_origin,
                        glyph_origin.x - underline_origin.x,
                        underline_style,
                    );
                    underline_origin.x = origin.x;
                    underline_origin.y += line_height;
                }
                if let Some((strikethrough_origin, strikethrough_style)) = current_strikethrough.as_mut() {
                    if glyph_origin.x == strikethrough_origin.x {
                        strikethrough_origin.x -= max_glyph_size.width.half();
                    };
                    painter.paint_strikethrough(
                        *strikethrough_origin,
                        glyph_origin.x - strikethrough_origin.x,
                        strikethrough_style,
                    );
                    strikethrough_origin.x = origin.x;
                    strikethrough_origin.y += line_height;
                }

                glyph_origin.x = aligned_origin_x(
                    origin,
                    align_width.unwrap_or(layout.width),
                    glyph.position.x,
                    &align,
                    layout,
                    wraps.peek(),
                );
                glyph_origin.y += line_height;
            }
            prev_glyph_position = glyph.position;

            let mut finished_underline: Option<(Point<Pixels>, UnderlineStyle)> = None;
            let mut finished_strikethrough: Option<(Point<Pixels>, StrikethroughStyle)> = None;
            if glyph.index >= run_end {
                let mut style_run = decoration_runs.next();

                // Ignore style runs that apply to a partial glyph.
                while let Some(run) = style_run {
                    if glyph.index < run_end + (run.len as usize) {
                        break;
                    }
                    run_end += run.len as usize;
                    style_run = decoration_runs.next();
                }

                if let Some(style_run) = style_run {
                    if let Some((_, underline_style)) = &mut current_underline
                        && style_run.underline.as_ref() != Some(underline_style)
                    {
                        finished_underline = current_underline.take();
                    }
                    if let Some(run_underline) = style_run.underline.as_ref() {
                        current_underline.get_or_insert((
                            point(
                                glyph_origin.x,
                                glyph_origin.y + baseline_offset.y + (layout.descent * 0.618),
                            ),
                            UnderlineStyle {
                                color: Some(run_underline.color.unwrap_or(style_run.color)),
                                thickness: run_underline.thickness,
                                wavy: run_underline.wavy,
                            },
                        ));
                    }
                    if let Some((_, strikethrough_style)) = &mut current_strikethrough
                        && style_run.strikethrough.as_ref() != Some(strikethrough_style)
                    {
                        finished_strikethrough = current_strikethrough.take();
                    }
                    if let Some(run_strikethrough) = style_run.strikethrough.as_ref() {
                        current_strikethrough.get_or_insert((
                            point(
                                glyph_origin.x,
                                glyph_origin.y + (((layout.ascent * 0.5) + baseline_offset.y) * 0.5),
                            ),
                            StrikethroughStyle {
                                color: Some(run_strikethrough.color.unwrap_or(style_run.color)),
                                thickness: run_strikethrough.thickness,
                            },
                        ));
                    }

                    run_end += style_run.len as usize;
                    color = style_run.color;
                } else {
                    run_end = layout.len;
                    finished_underline = current_underline.take();
                    finished_strikethrough = current_strikethrough.take();
                }
            }

            if let Some((mut underline_origin, underline_style)) = finished_underline {
                if underline_origin.x == glyph_origin.x {
                    underline_origin.x -= max_glyph_size.width.half();
                };
                painter.paint_underline(
                    underline_origin,
                    glyph_origin.x - underline_origin.x,
                    &underline_style,
                );
            }

            if let Some((mut strikethrough_origin, strikethrough_style)) = finished_strikethrough {
                if strikethrough_origin.x == glyph_origin.x {
                    strikethrough_origin.x -= max_glyph_size.width.half();
                };
                painter.paint_strikethrough(
                    strikethrough_origin,
                    glyph_origin.x - strikethrough_origin.x,
                    &strikethrough_style,
                );
            }

            let max_glyph_bounds = Bounds { origin: glyph_origin, size: max_glyph_size };

            if max_glyph_bounds.intersects(&content_mask) {
                if glyph.is_emoji {
                    painter.paint_emoji(
                        glyph_origin + baseline_offset,
                        run.font_id,
                        glyph.id,
                        layout.font_size,
                    )?;
                } else {
                    painter.paint_glyph(
                        glyph_origin + baseline_offset,
                        run.font_id,
                        glyph.id,
                        layout.font_size,
                        color,
                    )?;
                }
            }
        }
    }

    let mut last_line_end_x = first_glyph_x + layout.width;
    if let Some(boundary) = wrap_boundaries.last() {
        let run = &layout.runs[boundary.run_ix];
        let glyph = &run.glyphs[boundary.glyph_ix];
        last_line_end_x -= glyph.position.x;
    }

    if let Some((mut underline_start, underline_style)) = current_underline.take() {
        if last_line_end_x == underline_start.x {
            underline_start.x -= max_glyph_size.width.half()
        };
        painter.paint_underline(underline_start, last_line_end_x - underline_start.x, &underline_style);
    }

    if let Some((mut strikethrough_start, strikethrough_style)) = current_strikethrough.take() {
        if last_line_end_x == strikethrough_start.x {
            strikethrough_start.x -= max_glyph_size.width.half()
        };
        painter.paint_strikethrough(
            strikethrough_start,
            last_line_end_x - strikethrough_start.x,
            &strikethrough_style,
        );
    }

    Ok(())
}

/// Paints the background colours of `layout`'s decoration runs, as `paint_line` lays the line out.
#[allow(clippy::too_many_arguments)]
pub fn paint_line_background(
    origin: Point<Pixels>,
    layout: &LineLayout,
    line_height: Pixels,
    align: TextAlign,
    align_width: Option<Pixels>,
    decoration_runs: &[DecorationRun],
    wrap_boundaries: &[WrapBoundary],
    text_system: &TextSystem,
    painter: &mut dyn GlyphPainter,
) -> Result<()> {
    let mut decoration_runs = decoration_runs.iter();
    let mut wraps = wrap_boundaries.iter().peekable();
    let mut run_end = 0;
    let mut current_background: Option<(Point<Pixels>, Hsla)> = None;
    let mut glyph_origin = point(
        aligned_origin_x(origin, align_width.unwrap_or(layout.width), px(0.0), &align, layout, wraps.peek()),
        origin.y,
    );
    let mut prev_glyph_position = Point::default();
    let mut max_glyph_size = size(px(0.), px(0.));
    for (run_ix, run) in layout.runs.iter().enumerate() {
        max_glyph_size = text_system.bounding_box(run.font_id, layout.font_size).size;

        for (glyph_ix, glyph) in run.glyphs.iter().enumerate() {
            glyph_origin.x += glyph.position.x - prev_glyph_position.x;

            if wraps.peek() == Some(&&WrapBoundary { run_ix, glyph_ix }) {
                wraps.next();
                if let Some((background_origin, background_color)) = current_background.as_mut() {
                    if glyph_origin.x == background_origin.x {
                        background_origin.x -= max_glyph_size.width.half()
                    }
                    painter.paint_background(
                        Bounds {
                            origin: *background_origin,
                            size: size(glyph_origin.x - background_origin.x, line_height),
                        },
                        *background_color,
                    );
                    background_origin.x = origin.x;
                    background_origin.y += line_height;
                }

                glyph_origin.x = aligned_origin_x(
                    origin,
                    align_width.unwrap_or(layout.width),
                    glyph.position.x,
                    &align,
                    layout,
                    wraps.peek(),
                );
                glyph_origin.y += line_height;
            }
            prev_glyph_position = glyph.position;

            let mut finished_background: Option<(Point<Pixels>, Hsla)> = None;
            if glyph.index >= run_end {
                let mut style_run = decoration_runs.next();

                // Ignore style runs that apply to a partial glyph.
                while let Some(run) = style_run {
                    if glyph.index < run_end + (run.len as usize) {
                        break;
                    }
                    run_end += run.len as usize;
                    style_run = decoration_runs.next();
                }

                if let Some(style_run) = style_run {
                    if let Some((_, background_color)) = &mut current_background
                        && style_run.background_color.as_ref() != Some(background_color)
                    {
                        finished_background = current_background.take();
                    }
                    if let Some(run_background) = style_run.background_color {
                        current_background
                            .get_or_insert((point(glyph_origin.x, glyph_origin.y), run_background));
                    }
                    run_end += style_run.len as usize;
                } else {
                    run_end = layout.len;
                    finished_background = current_background.take();
                }
            }

            if let Some((mut background_origin, background_color)) = finished_background {
                // GPUI takes the width before nudging the origin.
                let width = glyph_origin.x - background_origin.x;
                if background_origin.x == glyph_origin.x {
                    background_origin.x -= max_glyph_size.width.half();
                };
                painter.paint_background(
                    Bounds { origin: background_origin, size: size(width, line_height) },
                    background_color,
                );
            }
        }
    }

    // GPUI measures the last line's end from `origin.x` here (not from the first glyph, as
    // `paint_line` does).
    let mut last_line_end_x = origin.x + layout.width;
    if let Some(boundary) = wrap_boundaries.last() {
        let run = &layout.runs[boundary.run_ix];
        let glyph = &run.glyphs[boundary.glyph_ix];
        last_line_end_x -= glyph.position.x;
    }

    if let Some((mut background_origin, background_color)) = current_background.take() {
        if last_line_end_x == background_origin.x {
            background_origin.x -= max_glyph_size.width.half()
        };
        painter.paint_background(
            Bounds {
                origin: background_origin,
                size: size(last_line_end_x - background_origin.x, line_height),
            },
            background_color,
        );
    }

    Ok(())
}

/// The x where a visual line starts: GPUI's alignment of the line's width (to the next wrap boundary,
/// or the end) within `align_width`.
fn aligned_origin_x(
    origin: Point<Pixels>,
    align_width: Pixels,
    last_glyph_x: Pixels,
    align: &TextAlign,
    layout: &LineLayout,
    wrap_boundary: Option<&&WrapBoundary>,
) -> Pixels {
    let end_of_line = if let Some(WrapBoundary { run_ix, glyph_ix }) = wrap_boundary {
        layout.runs[*run_ix].glyphs[*glyph_ix].position.x
    } else {
        layout.width
    };

    let line_width = end_of_line - last_glyph_x;

    match align {
        TextAlign::Left => origin.x,
        TextAlign::Center => (origin.x * 2.0 + align_width - line_width) / 2.0,
        TextAlign::Right => origin.x + align_width - line_width,
    }
}
