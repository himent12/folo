//! The line chart: dashed strokes, its element and its painter.

use std::cell::Cell;
use std::rc::Rc;

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, App, Bounds, Hsla, InteractiveElement, IntoElement, MouseMoveEvent, ParentElement,
    PathBuilder, Pixels, Point, StatefulInteractiveElement, Styled, Window, canvas, div, fill,
    linear_color_stop, linear_gradient, point, px, size,
};

use super::axis::{
    axis_pow, even_ticks, format_step, int_ticks, nice_step, pow2_stride, pow2_ticks_every, superscript,
    ticks,
};
use super::cursor::{chip_label, cursor_x, nearest};
use super::ease::ease;
use super::smooth::{cached_envelope, memo, smoothed};
use super::spec::{LineChart, Marker};
use super::text::{paint_text, text_width};
use crate::theme::{Theme, alpha, ty};

/// Height of one marker-label lane above a plot.
const LANE: f32 = 16.;

/// A label's start, kept between `lo` and `hi` (its last fitting start). Unlike `clamp`, a span
/// narrower than the label (a chart mid-resize, laid out at a few px) pins it to `lo` rather
/// than panicking.
fn clamp_in(v: Pixels, lo: Pixels, hi: Pixels) -> Pixels {
    v.min(hi).max(lo)
}

/// Dash patterns (on, off) in px.
const DASHED: &[f32] = &[4., 3.];
const DOTTED: &[f32] = &[1.5, 3.];
/// Long dashes and dash-dot, for line sets beyond solid / dashed / dotted (the PR curves).
pub const LONG_DASH: &[f32] = &[9., 3.];
pub const DASH_DOT: &[f32] = &[6., 2.5, 1.5, 2.5];

pub(super) fn stroke_path(window: &mut Window, pts: &[Point<Pixels>], width: f32, color: Hsla, dash: &[f32]) {
    if pts.len() < 2 {
        return;
    }
    let mut b = PathBuilder::stroke(px(width));
    if !dash.is_empty() {
        let d: Vec<Pixels> = dash.iter().map(|v| px(*v)).collect();
        b = b.dash_array(&d);
    }
    b.move_to(pts[0]);
    for p in &pts[1..] {
        b.line_to(*p);
    }
    if let Ok(path) = b.build() {
        window.paint_path(path, color);
    }
}

/// A line chart that fills its width.
pub fn line_chart(t: &Theme, spec: LineChart) -> AnyElement {
    let t = *t;
    let plot_cell: Rc<Cell<Option<Bounds<Pixels>>>> = Rc::new(Cell::new(None));
    let cell_paint = plot_cell.clone();
    let on_hover = spec.on_hover.clone();
    let on_leave = spec.on_hover.clone();
    let id = spec.id.clone();
    let height = spec.height;
    div()
        .id(folo_ui::ElementId::Name(id.clone()))
        .w_full()
        .h(px(height))
        .flex_none()
        .when_some(on_hover, |d, f| {
            let cell = plot_cell.clone();
            d.on_mouse_move(move |ev: &MouseMoveEvent, window, cx| {
                if let Some(b) = cell.get() {
                    let inside = b.contains(&ev.position)
                        || (ev.position.y >= b.top()
                            && ev.position.y <= b.bottom() + px(20.)
                            && ev.position.x >= b.left() - px(4.)
                            && ev.position.x <= b.right() + px(4.));
                    let frac = ((ev.position.x - b.left()) / b.size.width).clamp(0.0, 1.0);
                    f(inside.then_some(frac), window, cx);
                }
            })
        })
        .when_some(on_leave, |d, f| {
            d.on_hover(move |hovered, window, cx| {
                if !hovered {
                    f(None, window, cx)
                }
            })
        })
        .child(
            canvas(
                |_, _, _| (),
                move |bounds, _, window, cx| {
                    let mut spec = spec;
                    ease(&mut spec, window);
                    paint_line_chart(&t, &spec, bounds, &cell_paint, window, cx)
                },
            )
            .size_full(),
        )
        .into_any_element()
}

fn paint_line_chart(
    t: &Theme,
    spec: &LineChart,
    bounds: Bounds<Pixels>,
    plot_cell: &Cell<Option<Bounds<Pixels>>>,
    window: &mut Window,
    cx: &mut App,
) {
    // small: chart tick labels (and the axis's shared power), the one place design/README allows `xs`.
    let label_size = ty::XS.size;
    // Annotations (markers, the projected span, the cursor's value chip) are labels: 12 px.
    let annot_size = ty::SM.size;
    // Data extents.
    let mut x0 = f32::INFINITY;
    let mut x1 = f32::NEG_INFINITY;
    let mut y0 = f32::INFINITY;
    let mut y1 = f32::NEG_INFINITY;
    for s in &spec.series {
        // [(min x, min y), (max x, max y)], computed once per data revision.
        let e = memo(&s.points, [2, 0, 0, 0], |pts| {
            let mut lo = (f32::INFINITY, f32::INFINITY);
            let mut hi = (f32::NEG_INFINITY, f32::NEG_INFINITY);
            for &(x, y) in pts {
                if x.is_finite() && y.is_finite() {
                    lo = (lo.0.min(x), lo.1.min(y));
                    hi = (hi.0.max(x), hi.1.max(y));
                }
            }
            vec![lo, hi]
        });
        x0 = x0.min(e[0].0);
        y0 = y0.min(e[0].1);
        x1 = x1.max(e[1].0);
        y1 = y1.max(e[1].1);
    }
    if let Some((a, b)) = spec.x_range {
        x0 = a;
        x1 = b;
    }
    let has_data = x0.is_finite() && y0.is_finite();
    if !has_data {
        x0 = 0.0;
        x1 = 1.0;
        y0 = 0.0;
        y1 = 1.0;
    }
    if let Some((a, b)) = spec.y_range {
        y0 = a;
        y1 = b;
    } else {
        let pad = ((y1 - y0) * 0.08).max(1e-6);
        y0 = if y0 >= 0.0 && y0 - pad < 0.0 { 0.0 } else { y0 - pad };
        y1 += pad;
        if !spec.log_y {
            let step = nice_step(y1 - y0, spec.y_ticks);
            y1 = (y1 / step).ceil() * step;
            if y0 != 0.0 {
                y0 = (y0 / step).floor() * step;
            }
        }
    }
    if x1 <= x0 {
        x1 = x0 + 1.0;
    }
    if y1 <= y0 {
        y1 = y0 + 1.0;
    }
    // Log axes tick at powers of two (0.5 1 2 4 8): a factor of two reads the same everywhere.
    // Log ticks thin out to one label per 22 px of chart height (short charts keep them apart),
    // stepping down from the top power so the steps stay even and the domain stays the data's.
    let fit = ((f32::from(bounds.size.height) - 40.) / 22.).floor().max(2.) as usize;
    let stride = pow2_stride(y0, y1, fit);
    let (ly0, ly1) = if spec.log_y { (y0.max(1e-6).ln(), y1.max(1e-6).ln()) } else { (y0, y1) };
    let yticks = if let Some(v) = &spec.y_tick_values {
        v.clone()
    } else if spec.log_y {
        pow2_ticks_every(y0, y1, stride)
    } else {
        ticks(y0, y1, spec.y_ticks)
    };
    let ystep = nice_step(y1 - y0, spec.y_ticks);
    // Tiny values share one power of ten, shown once ("0 5 10" and "× 10⁻³", design/README).
    let ypow = (spec.y_fmt.is_none() && !spec.log_y).then(|| axis_pow(y0, y1, spec.y_ticks)).flatten();
    let ylabels: Vec<String> = yticks
        .iter()
        .map(|v| match (spec.y_fmt, ypow) {
            // Powers of two read as plain numbers: 8, 4, 2, 1, 0.5, 0.25.
            (None, _) if spec.log_y => format!("{v}"),
            (Some(f), _) => f(*v),
            (None, Some(p)) => {
                let k = 10f32.powi(-p);
                format_step(*v * k, ystep * k)
            }
            (None, None) => format_step(*v, ystep),
        })
        .collect();
    // A shared minimum gutter keeps plots in a grid aligned.
    let label_w = ylabels.iter().map(|l| text_width(window, l, label_size)).fold(px(30.), |a, b| a.max(b));
    let left = bounds.left() + label_w + px(10.);
    let plot_w = bounds.right() - left - px(2.);
    let sx0 = |x: f32| left + plot_w * ((x - x0) / (x1 - x0));
    // Marker labels live in a band above the plot, in lanes so they never overlap each other, and
    // the data never runs through them.
    let mut marker_layout: Vec<(Pixels, Pixels, usize, &Marker)> = Vec::new();
    let ms: Vec<&Marker> = spec.markers.iter().filter(|m| m.x >= x0 && m.x <= x1).collect();
    // Placed in the order given: the first marker takes the top lane.
    for m in ms {
        let x = sx0(m.x);
        let w = text_width(window, &m.label, annot_size) + px(8.);
        // Right of the line, or left of it near the right edge; on a collision try the other side
        // before opening a second lane.
        let right = x + px(2.);
        let left_side = x - px(2.) - w;
        let sides = if right + w > left + plot_w { [left_side, right] } else { [right, left_side] };
        let free = |lx: Pixels, lane: usize| {
            lx >= left - px(4.)
                && lx + w <= left + plot_w + px(4.)
                && !marker_layout
                    .iter()
                    .any(|(ox, ow, ol, _)| *ol == lane && *ox < lx + w + px(6.) && lx < *ox + *ow + px(6.))
        };
        // Labels stay on their line: the next lane down before drifting away from it.
        // A pinned lane first; on a collision the other lanes, so labels never overdraw.
        let order: Vec<usize> = match m.lane {
            Some(l) => std::iter::once(l).chain((0..3).filter(move |&x| x != l)).collect(),
            None => (0..3).collect(),
        };
        let spots: Vec<(Pixels, usize)> =
            order.iter().flat_map(|&lane| sides.into_iter().map(move |lx| (lx, lane))).collect();
        let fallback = (sides[0], m.lane.unwrap_or(2));
        let (lx, lane) = spots.into_iter().find(|&(lx, lane)| free(lx, lane)).unwrap_or(fallback);
        marker_layout.push((lx, w, lane, m));
    }
    let band = marker_layout.iter().map(|l| l.2 + 1).max().unwrap_or(0).max(spec.min_lanes) as f32 * LANE;
    let (top_pad, bottom_pad) = if spec.axis_titles.is_some() { (26. + band, 40.) } else { (6. + band, 22.) };
    let plot = Bounds::new(
        point(left, bounds.top() + px(top_pad)),
        size(plot_w, bounds.size.height - px(top_pad) - px(bottom_pad)),
    );
    if let Some((xt, yt)) = spec.axis_titles {
        paint_text(window, cx, yt, point(bounds.left(), bounds.top()), t.text_tertiary, ty::SM.size, false);
        let w = text_width(window, xt, ty::SM.size);
        paint_text(
            window,
            cx,
            xt,
            point(plot.left() + (plot.size.width - w) / 2., plot.bottom() + px(22.)),
            t.text_tertiary,
            ty::SM.size,
            false,
        );
    }
    plot_cell.set(Some(plot));
    let sx = |x: f32| plot.left() + plot.size.width * ((x - x0) / (x1 - x0));
    let sy = |y: f32| {
        let v = if spec.log_y { y.max(1e-6).ln() } else { y };
        plot.bottom() - plot.size.height * ((v - ly0) / (ly1 - ly0)).clamp(-0.05, 1.05)
    };

    // Grid + y labels.
    if let Some(p) = ypow.filter(|_| spec.pow_caption) {
        // The shared factor, once, above the axis labels.
        let f = format!("× 10{}", superscript(p));
        let w = text_width(window, &f, label_size);
        paint_text(
            window,
            cx,
            &f,
            point(plot.left() - px(8.) - w, plot.top() - px(label_size + 10.)),
            t.chart_label,
            label_size,
            false,
        );
    }
    for (v, l) in yticks.iter().zip(&ylabels) {
        let y = sy(*v);
        window.paint_quad(fill(
            Bounds::new(point(plot.left(), y), size(plot.size.width, px(1.))),
            t.chart_grid,
        ));
        paint_text(
            window,
            cx,
            l,
            point(plot.left() - px(8.), y - px(label_size / 2. + 2.)),
            t.chart_label,
            label_size,
            true,
        );
    }
    // Baseline axis.
    window.paint_quad(fill(
        Bounds::new(point(plot.left(), plot.bottom()), size(plot.size.width, px(1.))),
        t.chart_axis,
    ));
    // X ticks.
    // Remaining epochs read as planned, not empty: a faint tint and the time left.
    if let Some((from, label)) = &spec.future
        && *from < x1
    {
        // "Now" is the latest point drawn (the epoch in progress has no point yet): the tint starts
        // there, behind a 1 px rule, so no dark band opens between the data and the plan.
        let now =
            spec.series.iter().filter_map(|s| s.points.last()).map(|p| p.0).fold(f32::NEG_INFINITY, f32::max);
        // The data can run ahead of `from` (a history fetched after the run snapshot): the rule
        // follows the data, within the plot.
        let from = if now.is_finite() { now.clamp(x0, x1) } else { *from };
        let fx = sx(from).max(plot.left());
        window.paint_quad(fill(
            Bounds::new(point(fx, plot.top()), size(plot.right() - fx, plot.size.height)),
            alpha(t.text_primary, 0.025),
        ));
        window.paint_quad(fill(
            Bounds::new(point(fx, plot.top()), size(px(1.), plot.size.height)),
            t.line_default,
        ));
        if !label.is_empty() {
            let w = text_width(window, label, annot_size);
            // At the start of the span (where "now" is): in the marker band, like the other
            // annotations, when the chart has one and lane 0 is free there; else inside the plot.
            let lx = (fx + px(6.)).min(plot.right() - w - px(6.));
            let in_band = band > 0.
                && !marker_layout
                    .iter()
                    .any(|(ox, ow, lane, _)| *lane == 0 && *ox < lx + w + px(6.) && lx < *ox + *ow + px(6.));
            let y = if in_band { plot.top() - px(band) } else { plot.top() + px(4.) };
            paint_text(window, cx, label, point(lx, y), t.text_tertiary, annot_size, false);
        }
    }
    // Linked charts pass one tick count, so every chart in a group shows the same x ticks.
    let max_x = spec.x_ticks.unwrap_or(((plot.size.width / px(90.)) as usize).clamp(2, 8));
    let xt = if spec.x_int { int_ticks(x0, x1, max_x) } else { even_ticks(x0, x1, max_x) };
    // Where the cursor chip will sit (legend-driven charts), so tick labels give way to it.
    // The chip's real span (same label and clamping as where it's painted).
    let chip_span = (!spec.readout && has_data)
        .then_some(spec.hover)
        .flatten()
        .and_then(|f| cursor_x(&spec.series, x0 + (x1 - x0) * f, x1 - x0))
        .map(|v| {
            let w = text_width(window, &chip_label(spec, v), label_size) + px(10.);
            let left = clamp_in(sx(v) - w / 2., plot.left() - px(4.), plot.right() - w);
            (left, left + w)
        });
    for v in &xt {
        let x = sx(*v);
        window.paint_quad(fill(Bounds::new(point(x, plot.bottom()), size(px(1.), px(4.))), t.chart_axis));
        let l = (spec.x_fmt)(*v);
        let w = text_width(window, &l, label_size);
        // Centred on its tick (the end labels too: "0.00" / "1.00" flush with the tick read as
        // belonging beside it); only the chart's own edges push a label inward.
        let lx = clamp_in(x - w / 2., bounds.left(), bounds.right() - w);
        if chip_span.is_some_and(|(cl, cr)| lx < cr + px(4.) && lx + w > cl - px(4.)) {
            continue;
        }
        paint_text(window, cx, &l, point(lx, plot.bottom() + px(6.)), t.chart_label, label_size, false);
    }

    // Markers: dashed lines under the data, labels in the band above the plot.
    let mut labels: Vec<(Bounds<Pixels>, &Marker)> = Vec::new();
    for (lx, w, lane, m) in &marker_layout {
        let x = sx(m.x);
        stroke_path(
            window,
            &[point(x, plot.top()), point(x, plot.bottom())],
            1.0,
            alpha(m.color, 0.5),
            DASHED,
        );
        let r = Bounds::new(
            point(*lx, plot.top() - px(band) - px(1.) + px(*lane as f32 * LANE)),
            size(*w, px(15.)),
        );
        labels.push((r, m));
    }
    let columns = (plot.size.width / px(1.)).max(1.0) as usize;
    window.with_content_mask(
        Some(folo_ui::ContentMask {
            bounds: Bounds::new(
                point(plot.left() - px(4.), plot.top() - px(4.)),
                size(plot.size.width + px(8.), plot.size.height + px(5.)),
            ),
        }),
        |window| {
            // Live endpoint dots already drawn: one within 3 px of another is skipped.
            let mut live_dots: Vec<Point<Pixels>> = Vec::new();
            for s in &spec.series {
                if s.points.is_empty() {
                    continue;
                }
                let main = if s.smooth > 0.0 {
                    // Raw data faint underneath, smoothed on top.
                    let raw = cached_envelope(&s.points, x0, x1, columns);
                    let pts: Vec<Point<Pixels>> = raw.iter().map(|&(x, y)| point(sx(x), sy(y))).collect();
                    // Quiet enough that the smoothed lines' dash patterns carry identity.
                    stroke_path(window, &pts, 1.0, alpha(s.color, 0.05), &[]);
                    cached_envelope(&smoothed(&s.points, s.smooth), x0, x1, columns / 2 + 1)
                } else {
                    cached_envelope(&s.points, x0, x1, columns)
                };
                let pts: Vec<Point<Pixels>> = main.iter().map(|&(x, y)| point(sx(x), sy(y))).collect();
                if s.area && pts.len() > 1 {
                    let mut b = PathBuilder::fill();
                    b.move_to(point(pts[0].x, plot.bottom()));
                    for p in &pts {
                        b.line_to(*p);
                    }
                    b.line_to(point(pts[pts.len() - 1].x, plot.bottom()));
                    b.close();
                    if let Ok(path) = b.build() {
                        let top = alpha(s.color, t.chart_area_opacity * 1.6);
                        let bottom = alpha(s.color, 0.0);
                        window.paint_path(
                            path,
                            linear_gradient(180., linear_color_stop(top, 0.), linear_color_stop(bottom, 1.)),
                        );
                    }
                }
                let dash = if let Some(p) = s.pattern {
                    p
                } else if s.dotted {
                    DOTTED
                } else if s.dashed {
                    DASHED
                } else {
                    &[]
                };
                stroke_path(window, &pts, s.width, s.color, dash);
                if s.points.len() < 5 {
                    for &(x, y) in s.points.iter() {
                        let c = point(sx(x), sy(y));
                        window.paint_quad(
                            fill(Bounds::centered_at(c, size(px(5.), px(5.))), s.color).corner_radii(px(2.5)),
                        );
                    }
                }
                // While the cursor inspects, its dots are the markers; latest-point halos return on exit.
                if s.live
                    && spec.hover.is_none()
                    && let Some(&(x, y)) = main.last()
                    && !live_dots.iter().any(|d| {
                        let c = point(sx(x), sy(y));
                        // Only a dot that would sit on another is dropped; near ones both show.
                        (d.x - c.x).abs() < px(3.) && (d.y - c.y).abs() < px(3.)
                    })
                {
                    let c = point(sx(x), sy(y));
                    live_dots.push(c);
                    window.paint_quad(
                        fill(Bounds::centered_at(c, size(px(10.), px(10.))), alpha(s.color, 0.22))
                            .corner_radii(px(5.)),
                    );
                    window.paint_quad(
                        fill(Bounds::centered_at(c, size(px(5.), px(5.))), s.color).corner_radii(px(2.5)),
                    );
                }
            }
        },
    );

    // Dots that would overlap are nudged apart (8 px between centres), so none hides another.
    let mut placed: Vec<Point<Pixels>> = Vec::new();
    for &(x, y, c) in &spec.dots {
        if x >= x0 && x <= x1 {
            let at = point(sx(x), sy(y));
            // Its own spot if clear, else the first clear one on rings of 8, 12 and 16 px around
            // it (eight directions each, deterministic), else its own spot after all.
            let clear = |p: Point<Pixels>| {
                placed.iter().all(|q| {
                    let (dx, dy) = (f32::from(p.x - q.x), f32::from(p.y - q.y));
                    dx * dx + dy * dy >= 63.9
                })
            };
            let p = std::iter::once(at)
                .chain([8f32, 12., 16.].into_iter().flat_map(|r| {
                    (0..8).map(move |k| {
                        let a = k as f32 * std::f32::consts::FRAC_PI_4;
                        point(at.x + px(r * a.cos()), at.y + px(r * a.sin()))
                    })
                }))
                .find(|&p| clear(p))
                .unwrap_or(at);
            placed.push(p);
            window.paint_quad(
                fill(Bounds::centered_at(p, size(px(10.), px(10.))), t.bg_panel).corner_radii(px(5.)),
            );
            window.paint_quad(fill(Bounds::centered_at(p, size(px(7.), px(7.))), c).corner_radii(px(3.5)));
        }
    }
    for (r, m) in &labels {
        window.paint_quad(fill(*r, t.bg_panel).corner_radii(px(3.)));
        paint_text(
            window,
            cx,
            &m.label,
            point(r.left() + px(4.), r.top() + px(1.)),
            m.color,
            annot_size,
            false,
        );
    }

    // Hover cursor + readout.
    let Some(frac) = spec.hover else { return };
    if !has_data {
        return;
    }
    let hx = x0 + (x1 - x0) * frac;
    let mut rows: Vec<(Hsla, String, String)> = Vec::new();
    let mut point_ys: Vec<Pixels> = Vec::new();
    let mut snap_x = None;
    for s in &spec.series {
        let data = if s.smooth > 0.0 { smoothed(&s.points, s.smooth) } else { s.points.clone() };
        if let Some(i) = nearest(&data, hx) {
            let (x, y) = data[i];
            if (x - hx).abs() > (x1 - x0) * 0.06 && s.points.len() > 1 {
                continue;
            }
            snap_x.get_or_insert(x);
            let c = point(sx(x), sy(y));
            point_ys.push(c.y);
            window.paint_quad(
                fill(Bounds::centered_at(c, size(px(9.), px(9.))), t.bg_overlay).corner_radii(px(4.5)),
            );
            window
                .paint_quad(fill(Bounds::centered_at(c, size(px(6.), px(6.))), s.color).corner_radii(px(3.)));
            rows.push((
                s.color,
                s.name.to_string(),
                spec.value_fmt.map_or_else(|| format_step(y, ystep / 10.), |f| f(y)),
            ));
        }
    }
    let Some(snap) = snap_x else { return };
    let cx_px = sx(snap);
    window.paint_quad(fill(
        Bounds::new(point(cx_px, plot.top()), size(px(1.), plot.size.height)),
        t.chart_cursor,
    ));
    if !spec.readout {
        // The values are in the header legend; the axis says where the cursor is.
        let l = chip_label(spec, snap);
        let w = text_width(window, &l, annot_size) + px(10.);
        let chip = Bounds::new(
            point(clamp_in(cx_px - w / 2., plot.left() - px(4.), plot.right() - w), plot.bottom() + px(3.)),
            size(w, px(16.)),
        );
        window.paint_quad(fill(chip, t.text_primary).corner_radii(px(3.)));
        paint_text(
            window,
            cx,
            &l,
            point(chip.left() + px(5.), chip.top() + px(1.)),
            t.bg_panel,
            annot_size,
            false,
        );
        return;
    }
    // Readout box.
    let smoothed_any = spec.series.iter().any(|s| s.smooth > 0.0);
    let title =
        format!("{} {}{}", spec.x_name, (spec.x_fmt)(snap), if smoothed_any { " · smoothed" } else { "" })
            .trim()
            .to_string();
    let fs = ty::SM.size;
    let title_w = text_width(window, &title, fs);
    let name_w = rows.iter().map(|r| text_width(window, &r.1, fs)).fold(title_w, |a, b| a.max(b));
    let val_w = rows.iter().map(|r| text_width(window, &r.2, fs)).fold(px(0.), |a, b| a.max(b));
    let w = px(10.) + px(12.) + name_w + px(16.) + val_w + px(10.);
    let h = px(8.) + px(16.) * (rows.len() as f32 + 1.) + px(6.);
    // Right of the cursor in the left part of the plot, left of it beyond 60 %, so the readout never
    // covers the newest data at the series' end.
    let data_end =
        spec.series.iter().filter_map(|s| s.points.last().map(|p| p.0)).fold(f32::NEG_INFINITY, f32::max);
    let extent_right = if data_end.is_finite() { sx(data_end).min(plot.right()) } else { plot.right() };
    let flip = (cx_px > plot.left() + (extent_right - plot.left()) * 0.6
        && cx_px - px(12.) - w >= bounds.left())
        || cx_px + px(12.) + w > plot.right();
    // Beside the cursor (preferred side first), at the top or bottom of the plot: whichever spot
    // covers the least data, and isn't level with the highlighted points.
    let left_x = (cx_px - px(12.) - w).max(bounds.left());
    let right_x = cx_px + px(12.);
    let fits_right = right_x + w <= plot.right();
    let fits_left = cx_px - px(12.) - w >= bounds.left();
    let sides: Vec<Pixels> = match (flip, fits_left, fits_right) {
        (true, _, true) | (false, true, true) => {
            if flip {
                vec![left_x, right_x]
            } else {
                vec![right_x, left_x]
            }
        }
        (_, true, false) => vec![left_x],
        _ => vec![if flip { left_x } else { right_x }],
    };
    let covered = |bx: Pixels, by: Pixels| {
        let r = Bounds::new(point(bx - px(4.), by - px(4.)), size(w + px(8.), h + px(8.)));
        spec.series
            .iter()
            .map(|s| {
                cached_envelope(&s.points, x0, x1, columns)
                    .iter()
                    .filter(|&&(x, y)| r.contains(&point(sx(x), sy(y))))
                    .count()
            })
            .sum::<usize>()
            + point_ys.iter().filter(|&&y| y > r.top() && y < r.bottom()).count() * 50
    };
    let top_y = plot.top() + px(4.);
    let bottom_y = plot.bottom() - h - px(4.);
    let (bx, by) = sides
        .iter()
        .flat_map(|&x| [(x, top_y), (x, bottom_y)])
        .min_by_key(|&(x, y)| covered(x, y))
        .unwrap_or((sides[0], top_y));
    let rect = Bounds::new(point(bx, by), size(w, h));
    window.paint_quad(folo_ui::quad(
        rect,
        px(6.),
        t.bg_overlay,
        px(1.),
        t.line_strong,
        folo_ui::BorderStyle::Solid,
    ));
    paint_text(window, cx, &title, point(bx + px(10.), by + px(6.)), t.text_tertiary, fs, false);
    for (i, (color, name, value)) in rows.iter().enumerate() {
        let y = by + px(6.) + px(16.) * (i as f32 + 1.);
        window.paint_quad(
            fill(Bounds::new(point(bx + px(10.), y + px(4.)), size(px(8.), px(8.))), *color)
                .corner_radii(px(2.)),
        );
        paint_text(window, cx, name, point(bx + px(24.), y), t.text_secondary, fs, false);
        paint_text(window, cx, value, point(bx + w - px(10.), y), t.text_primary, fs, true);
    }
}
