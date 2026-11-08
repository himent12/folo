//! Charts painted with GPUI canvas primitives: line charts (raw + smoothed series, gradient
//! areas, markers, hover crosshair with a readout), sparklines and histograms.
//!
//! Charts are plain functions of their data. Hover state is owned by the caller: the chart reports
//! the pointer's position as a fraction of the plot width through `on_hover` and draws the cursor
//! at the `hover` fraction it is given, so several charts can share one cursor.

use std::cell::Cell;
use std::rc::Rc;
use std::sync::Arc;

use gpui::prelude::FluentBuilder;
use gpui::{
    AnyElement, App, Bounds, Hsla, InteractiveElement, IntoElement, MouseMoveEvent, ParentElement,
    PathBuilder, Pixels, Point, SharedString, StatefulInteractiveElement, Styled, TextRun, Window, canvas,
    div, fill, linear_color_stop, linear_gradient, point, px, size,
};

use crate::theme::{MONO, Theme, alpha, ty};

pub type HoverFn = Rc<dyn Fn(Option<f32>, &mut Window, &mut App)>;

#[derive(Clone)]
pub struct Series {
    pub name: SharedString,
    pub color: Hsla,
    /// (x, y), sorted by x.
    pub points: Arc<Vec<(f32, f32)>>,
    /// Exponential smoothing weight (0 = none). Raw data is then drawn faint underneath.
    pub smooth: f32,
    pub area: bool,
    pub width: f32,
    pub dashed: bool,
    /// Draw a live dot with a halo on the last point.
    pub live: bool,
}

impl Series {
    pub fn new(name: impl Into<SharedString>, color: Hsla, points: Arc<Vec<(f32, f32)>>) -> Self {
        Self {
            name: name.into(),
            color,
            points,
            smooth: 0.0,
            area: false,
            width: 1.5,
            dashed: false,
            live: false,
        }
    }

    pub fn smooth(mut self, w: f32) -> Self {
        self.smooth = w;
        self
    }

    pub fn area(mut self) -> Self {
        self.area = true;
        self
    }

    pub fn live(mut self, live: bool) -> Self {
        self.live = live;
        self
    }

    pub fn width(mut self, w: f32) -> Self {
        self.width = w;
        self
    }
}

#[derive(Clone)]
pub struct Marker {
    pub x: f32,
    pub label: SharedString,
    pub color: Hsla,
}

pub struct LineChart {
    pub id: SharedString,
    pub series: Vec<Series>,
    pub x_range: Option<(f32, f32)>,
    pub y_range: Option<(f32, f32)>,
    /// Tick label format; `None` picks decimals from the tick step so labels line up.
    pub y_fmt: Option<fn(f32) -> String>,
    pub x_fmt: fn(f32) -> String,
    /// Readout values use this format (defaults to `y_fmt`).
    pub value_fmt: Option<fn(f32) -> String>,
    pub x_name: &'static str,
    pub markers: Vec<Marker>,
    pub hover: Option<f32>,
    pub on_hover: Option<HoverFn>,
    pub height: f32,
    /// y ticks to aim for.
    pub y_ticks: usize,
    pub log_y: bool,
}

impl LineChart {
    pub fn new(id: impl Into<SharedString>, series: Vec<Series>) -> Self {
        Self {
            id: id.into(),
            series,
            x_range: None,
            y_range: None,
            y_fmt: None,
            x_fmt: |v| format_tick(v),
            value_fmt: None,
            x_name: "",
            markers: Vec::new(),
            hover: None,
            on_hover: None,
            height: 180.,
            y_ticks: 4,
            log_y: false,
        }
    }
}

/// Round tick spacing near `range / n`: 1, 2, 2.5 or 5 × 10^k.
pub fn nice_step(range: f32, n: usize) -> f32 {
    if range <= 0.0 || !range.is_finite() {
        return 1.0;
    }
    let raw = range / n.max(1) as f32;
    let mag = 10f32.powf(raw.log10().floor());
    let r = raw / mag;
    let m = if r <= 1.0 {
        1.0
    } else if r <= 2.0 {
        2.0
    } else if r <= 2.5 {
        2.5
    } else if r <= 5.0 {
        5.0
    } else {
        10.0
    };
    m * mag
}

pub fn ticks(lo: f32, hi: f32, n: usize) -> Vec<f32> {
    let step = nice_step(hi - lo, n);
    let mut v = (lo / step).ceil() * step;
    let mut out = Vec::new();
    while v <= hi + step * 1e-3 && out.len() < 20 {
        out.push(if v.abs() < step * 1e-6 { 0.0 } else { v });
        v += step;
    }
    out
}

/// Format `v` with the decimals a tick step of `step` needs (0.25 → 2, 0.5 → 1, 10 → 0).
pub fn format_step(v: f32, step: f32) -> String {
    if step.abs() >= 1000.0 && v.abs() >= 1000.0 {
        return format_tick(v);
    }
    let mut d = 0usize;
    let mut s = step.abs();
    while d < 6 && (s - s.round()).abs() > 1e-3 * s.max(1e-9) {
        s *= 10.0;
        d += 1;
    }
    format!("{v:.d$}")
}

pub fn format_tick(v: f32) -> String {
    let a = v.abs();
    if a == 0.0 {
        "0".into()
    } else if a >= 10_000.0 {
        format!("{:.0}k", v / 1000.0)
    } else if a >= 1000.0 {
        format!("{:.1}k", v / 1000.0)
    } else if a >= 10.0 {
        format!("{v:.0}")
    } else if a >= 1.0 {
        format!("{v:.1}")
    } else if a >= 0.01 {
        format!("{v:.2}")
    } else {
        format!("{v:.0e}")
    }
}

/// EMA smoothing with TensorBoard-style debiasing so the start isn't pulled toward zero.
pub fn ema(points: &[(f32, f32)], w: f32) -> Vec<(f32, f32)> {
    let mut last = 0.0;
    let mut out = Vec::with_capacity(points.len());
    for (i, &(x, y)) in points.iter().enumerate() {
        last = last * w + (1.0 - w) * y;
        let debias = 1.0 - w.powi(i as i32 + 1);
        out.push((x, last / debias));
    }
    out
}

/// Reduce to at most ~2 points per pixel column, keeping each column's min and max so spikes
/// survive downsampling.
fn envelope(points: &[(f32, f32)], x0: f32, x1: f32, columns: usize) -> Vec<(f32, f32)> {
    if points.len() <= columns * 2 || x1 <= x0 {
        return points.to_vec();
    }
    let mut out = Vec::with_capacity(columns * 2);
    let span = x1 - x0;
    let mut i = 0;
    for c in 0..columns {
        let end = x0 + span * (c + 1) as f32 / columns as f32;
        let start = i;
        let (mut lo, mut hi) = ((0.0, f32::INFINITY), (0.0, f32::NEG_INFINITY));
        while i < points.len() && (points[i].0 <= end || c == columns - 1) {
            let p = points[i];
            if p.1 < lo.1 {
                lo = p;
            }
            if p.1 > hi.1 {
                hi = p;
            }
            i += 1;
        }
        if i > start {
            if lo.0 <= hi.0 {
                out.push(lo);
                if hi != lo {
                    out.push(hi);
                }
            } else {
                out.push(hi);
                out.push(lo);
            }
        }
    }
    out
}

type Memo = Vec<(std::sync::Weak<Vec<(f32, f32)>>, [u32; 4], Arc<Vec<(f32, f32)>>)>;

thread_local! {
    /// Derived series keyed by their source allocation (held weakly, so a freed and reused
    /// allocation can never alias) plus the derivation parameters.
    static MEMO: std::cell::RefCell<Memo> = const { std::cell::RefCell::new(Vec::new()) };
}

fn memo(
    src: &Arc<Vec<(f32, f32)>>,
    params: [u32; 4],
    f: impl FnOnce(&[(f32, f32)]) -> Vec<(f32, f32)>,
) -> Arc<Vec<(f32, f32)>> {
    MEMO.with(|m| {
        let mut m = m.borrow_mut();
        m.retain(|(w, _, _)| w.strong_count() > 0);
        if let Some((_, _, v)) =
            m.iter().find(|(w, p, _)| *p == params && w.upgrade().is_some_and(|a| Arc::ptr_eq(&a, src)))
        {
            return v.clone();
        }
        let v = Arc::new(f(src));
        if m.len() > 256 {
            m.remove(0);
        }
        m.push((Arc::downgrade(src), params, v.clone()));
        v
    })
}

/// The EMA of a series, computed once per data revision and weight.
pub fn smoothed(points: &Arc<Vec<(f32, f32)>>, w: f32) -> Arc<Vec<(f32, f32)>> {
    memo(points, [0, w.to_bits(), 0, 0], |p| ema(p, w))
}

/// The per-column envelope of a series for a plot width and x range, cached likewise.
fn cached_envelope(points: &Arc<Vec<(f32, f32)>>, x0: f32, x1: f32, columns: usize) -> Arc<Vec<(f32, f32)>> {
    memo(points, [1, x0.to_bits(), x1.to_bits(), columns as u32], |p| envelope(p, x0, x1, columns))
}

/// Index of the point nearest to `x` in a sorted series.
pub fn nearest(points: &[(f32, f32)], x: f32) -> Option<usize> {
    if points.is_empty() {
        return None;
    }
    let i = points.partition_point(|p| p.0 < x);
    if i == 0 {
        Some(0)
    } else if i >= points.len() {
        Some(points.len() - 1)
    } else if (points[i].0 - x).abs() < (x - points[i - 1].0).abs() {
        Some(i)
    } else {
        Some(i - 1)
    }
}

pub fn paint_text(
    window: &mut Window,
    cx: &mut App,
    text: &str,
    origin: Point<Pixels>,
    color: Hsla,
    size_px: f32,
    align_right: bool,
) -> Pixels {
    let run = TextRun {
        len: text.len(),
        font: gpui::Font {
            features: gpui::FontFeatures(Arc::new(vec![("tnum".into(), 1)])),
            ..gpui::font(MONO)
        },
        color,
        background_color: None,
        underline: None,
        strikethrough: None,
    };
    let line =
        window.text_system().shape_line(SharedString::from(text.to_string()), px(size_px), &[run], None);
    let w = line.width;
    let o = if align_right { point(origin.x - w, origin.y) } else { origin };
    let _ = line.paint(o, px(size_px + 4.), window, cx);
    w
}

fn text_width(window: &mut Window, text: &str, size_px: f32) -> Pixels {
    let run = TextRun {
        len: text.len(),
        font: gpui::font(MONO),
        color: gpui::black(),
        background_color: None,
        underline: None,
        strikethrough: None,
    };
    window.text_system().shape_line(SharedString::from(text.to_string()), px(size_px), &[run], None).width
}

fn stroke_path(window: &mut Window, pts: &[Point<Pixels>], width: f32, color: Hsla, dashed: bool) {
    if pts.len() < 2 {
        return;
    }
    let mut b = PathBuilder::stroke(px(width));
    if dashed {
        b = b.dash_array(&[px(4.), px(3.)]);
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
        .id(gpui::ElementId::Name(id.clone()))
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
                move |bounds, _, window, cx| paint_line_chart(&t, &spec, bounds, &cell_paint, window, cx),
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
    let label_size = ty::XS.size - 0.5;
    // Data extents.
    let mut x0 = f32::INFINITY;
    let mut x1 = f32::NEG_INFINITY;
    let mut y0 = f32::INFINITY;
    let mut y1 = f32::NEG_INFINITY;
    for s in &spec.series {
        for &(x, y) in s.points.iter() {
            if !x.is_finite() || !y.is_finite() {
                continue;
            }
            x0 = x0.min(x);
            x1 = x1.max(x);
            y0 = y0.min(y);
            y1 = y1.max(y);
        }
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
    }
    if x1 <= x0 {
        x1 = x0 + 1.0;
    }
    if y1 <= y0 {
        y1 = y0 + 1.0;
    }
    let (ly0, ly1) = if spec.log_y { (y0.max(1e-6).ln(), y1.max(1e-6).ln()) } else { (y0, y1) };
    let yticks = if spec.log_y { Vec::new() } else { ticks(y0, y1, spec.y_ticks) };
    let ystep = nice_step(y1 - y0, spec.y_ticks);
    let ylabels: Vec<String> =
        yticks.iter().map(|v| spec.y_fmt.map_or_else(|| format_step(*v, ystep), |f| f(*v))).collect();
    // A shared minimum gutter keeps plots in a grid aligned.
    let label_w = ylabels.iter().map(|l| text_width(window, l, label_size)).fold(px(28.), |a, b| a.max(b));
    let left = bounds.left() + label_w + px(10.);
    let plot = Bounds::new(
        point(left, bounds.top() + px(6.)),
        size(bounds.right() - left - px(2.), bounds.size.height - px(6.) - px(22.)),
    );
    plot_cell.set(Some(plot));
    let sx = |x: f32| plot.left() + plot.size.width * ((x - x0) / (x1 - x0));
    let sy = |y: f32| {
        let v = if spec.log_y { y.max(1e-6).ln() } else { y };
        plot.bottom() - plot.size.height * ((v - ly0) / (ly1 - ly0)).clamp(-0.05, 1.05)
    };

    // Grid + y labels.
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
    let xt = ticks(x0, x1, ((plot.size.width / px(90.)) as usize).clamp(2, 8));
    for v in &xt {
        let x = sx(*v);
        window.paint_quad(fill(Bounds::new(point(x, plot.bottom()), size(px(1.), px(4.))), t.chart_axis));
        let l = (spec.x_fmt)(*v);
        let w = text_width(window, &l, label_size);
        let lx = (x - w / 2.).clamp(plot.left() - px(4.), plot.right() - w);
        paint_text(window, cx, &l, point(lx, plot.bottom() + px(6.)), t.chart_label, label_size, false);
    }
    if !spec.x_name.is_empty() {
        let _ = spec.x_name;
    }

    // Markers (vertical, dashed) under the data.
    for m in &spec.markers {
        if m.x < x0 || m.x > x1 {
            continue;
        }
        let x = sx(m.x);
        stroke_path(
            window,
            &[point(x, plot.top()), point(x, plot.bottom())],
            1.0,
            alpha(m.color, 0.55),
            true,
        );
        let w = text_width(window, &m.label, label_size);
        let lx = if x + px(4.) + w > plot.right() { x - px(4.) - w } else { x + px(4.) };
        paint_text(window, cx, &m.label, point(lx, plot.top() + px(1.)), m.color, label_size, false);
    }

    let columns = (plot.size.width / px(1.)).max(1.0) as usize;
    window.with_content_mask(
        Some(gpui::ContentMask {
            bounds: Bounds::new(
                point(plot.left() - px(4.), plot.top() - px(4.)),
                size(plot.size.width + px(8.), plot.size.height + px(5.)),
            ),
        }),
        |window| {
            for s in &spec.series {
                if s.points.is_empty() {
                    continue;
                }
                let main = if s.smooth > 0.0 {
                    // Raw data faint underneath, smoothed on top.
                    let raw = cached_envelope(&s.points, x0, x1, columns);
                    let pts: Vec<Point<Pixels>> = raw.iter().map(|&(x, y)| point(sx(x), sy(y))).collect();
                    stroke_path(window, &pts, 1.0, alpha(s.color, 0.22), false);
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
                stroke_path(window, &pts, s.width, s.color, s.dashed);
                if s.live
                    && let Some(&(x, y)) = main.last()
                {
                    let c = point(sx(x), sy(y));
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

    // Hover cursor + readout.
    let Some(frac) = spec.hover else { return };
    if !has_data {
        return;
    }
    let hx = x0 + (x1 - x0) * frac;
    let mut rows: Vec<(Hsla, String, String)> = Vec::new();
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
    // Readout box.
    let title = format!("{} {}", spec.x_name, (spec.x_fmt)(snap)).trim().to_string();
    let fs = ty::XS.size;
    let title_w = text_width(window, &title, fs);
    let name_w = rows.iter().map(|r| text_width(window, &r.1, fs)).fold(title_w, |a, b| a.max(b));
    let val_w = rows.iter().map(|r| text_width(window, &r.2, fs)).fold(px(0.), |a, b| a.max(b));
    let w = px(10.) + px(12.) + name_w + px(16.) + val_w + px(10.);
    let h = px(8.) + px(16.) * (rows.len() as f32 + 1.) + px(6.);
    let bx = if cx_px + px(12.) + w > plot.right() { cx_px - px(12.) - w } else { cx_px + px(12.) };
    let bx = bx.max(bounds.left());
    let by = plot.top() + px(4.);
    let rect = Bounds::new(point(bx, by), size(w, h));
    window.paint_quad(gpui::quad(
        rect,
        px(6.),
        t.bg_overlay,
        px(1.),
        t.line_strong,
        gpui::BorderStyle::Solid,
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

/// A tiny axis-less trend line with an optional area and live end dot.
pub fn sparkline(t: &Theme, values: Arc<Vec<f32>>, color: Hsla, height: f32, live: bool) -> AnyElement {
    let t = *t;
    canvas(
        |_, _, _| (),
        move |b, _, window, _| {
            if values.len() < 2 {
                return;
            }
            let (lo, hi) =
                values.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &v| (lo.min(v), hi.max(v)));
            let span = (hi - lo).max(1e-6);
            let inset = px(3.);
            let w = b.size.width - inset * 2.;
            let h = b.size.height - inset * 2.;
            let n = values.len();
            let pts: Vec<Point<Pixels>> = values
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    point(
                        b.left() + inset + w * (i as f32 / (n - 1) as f32),
                        b.top() + inset + h * (1.0 - (v - lo) / span),
                    )
                })
                .collect();
            let mut a = PathBuilder::fill();
            a.move_to(point(pts[0].x, b.bottom()));
            for p in &pts {
                a.line_to(*p);
            }
            a.line_to(point(pts[n - 1].x, b.bottom()));
            a.close();
            if let Ok(path) = a.build() {
                window.paint_path(
                    path,
                    linear_gradient(
                        180.,
                        linear_color_stop(alpha(color, t.chart_area_opacity * 1.4), 0.),
                        linear_color_stop(alpha(color, 0.), 1.),
                    ),
                );
            }
            stroke_path(window, &pts, 1.25, color, false);
            if live {
                let c = pts[n - 1];
                window.paint_quad(
                    fill(Bounds::centered_at(c, size(px(4.), px(4.))), color).corner_radii(px(2.)),
                );
            }
        },
    )
    .w_full()
    .h(px(height))
    .into_any_element()
}

/// Vertical bars for a histogram; `labels` sit under the bars (one per bin).
pub fn histogram(
    t: &Theme,
    id: impl Into<SharedString>,
    counts: Arc<Vec<u32>>,
    labels: Arc<Vec<String>>,
    color: Hsla,
    height: f32,
    hover: Option<f32>,
    on_hover: Option<HoverFn>,
) -> AnyElement {
    let t = *t;
    let cell: Rc<Cell<Option<Bounds<Pixels>>>> = Rc::new(Cell::new(None));
    let paint_cell = cell.clone();
    let leave = on_hover.clone();
    div()
        .id(gpui::ElementId::Name(id.into()))
        .w_full()
        .h(px(height))
        .when_some(on_hover, |d, f| {
            let cell = cell.clone();
            d.on_mouse_move(move |ev: &MouseMoveEvent, window, cx| {
                if let Some(b) = cell.get() {
                    let frac = ((ev.position.x - b.left()) / b.size.width).clamp(0.0, 0.9999);
                    f(b.contains(&ev.position).then_some(frac), window, cx);
                }
            })
        })
        .when_some(leave, |d, f| {
            d.on_hover(move |h, window, cx| {
                if !h {
                    f(None, window, cx)
                }
            })
        })
        .child(
            canvas(
                |_, _, _| (),
                move |b, _, window, cx| {
                    let n = counts.len().max(1);
                    let max = counts.iter().copied().max().unwrap_or(1).max(1) as f32;
                    let label_h = px(18.);
                    let plot = Bounds::new(b.origin, size(b.size.width, b.size.height - label_h));
                    paint_cell.set(Some(plot));
                    let gap = px(4.);
                    let bw = (plot.size.width - gap * (n as f32 - 1.)) / n as f32;
                    let hovered = hover.map(|f| (f * n as f32) as usize);
                    window.paint_quad(fill(
                        Bounds::new(point(plot.left(), plot.bottom()), size(plot.size.width, px(1.))),
                        t.chart_axis,
                    ));
                    for (i, &c) in counts.iter().enumerate() {
                        let x = plot.left() + (bw + gap) * i as f32;
                        let h = (plot.size.height - px(16.)) * (c as f32 / max);
                        let h = if c > 0 { h.max(px(2.)) } else { px(0.) };
                        let hot = hovered == Some(i);
                        let col = if hovered.is_some() && !hot { alpha(color, 0.45) } else { color };
                        window.paint_quad(
                            fill(Bounds::new(point(x, plot.bottom() - h), size(bw, h)), col).corner_radii(
                                gpui::Corners {
                                    top_left: px(2.),
                                    top_right: px(2.),
                                    bottom_left: px(0.),
                                    bottom_right: px(0.),
                                },
                            ),
                        );
                        if let Some(l) = labels.get(i) {
                            let fs = ty::XS.size - 0.5;
                            let w = text_width(window, l, fs);
                            if w < bw + gap * 2. || i % 2 == 0 {
                                paint_text(
                                    window,
                                    cx,
                                    l,
                                    point(x + bw / 2. - w / 2., plot.bottom() + px(5.)),
                                    t.chart_label,
                                    fs,
                                    false,
                                );
                            }
                        }
                        if hot {
                            let s = crate::ui::int(c as u64);
                            let w = text_width(window, &s, ty::XS.size);
                            paint_text(
                                window,
                                cx,
                                &s,
                                point(x + bw / 2. - w / 2., plot.bottom() - h - px(15.)),
                                t.text_primary,
                                ty::XS.size,
                                false,
                            );
                        }
                    }
                },
            )
            .size_full(),
        )
        .into_any_element()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nice_steps() {
        assert_eq!(nice_step(1.0, 4), 0.25);
        assert_eq!(nice_step(100.0, 5), 20.0);
        assert_eq!(nice_step(7.0, 4), 2.0);
        let t = ticks(0.0, 1.0, 4);
        assert_eq!(t, vec![0.0, 0.25, 0.5, 0.75, 1.0]);
        assert_eq!(format_step(0.5, 0.25), "0.50");
        assert_eq!(format_step(2.0, 1.0), "2");
        assert_eq!(format_step(0.004, 0.002), "0.004");
        assert_eq!(format_step(1.5, 0.5), "1.5");
    }

    #[test]
    fn ema_is_debiased() {
        let pts: Vec<(f32, f32)> = (0..10).map(|i| (i as f32, 5.0)).collect();
        assert!(ema(&pts, 0.9).iter().all(|p| (p.1 - 5.0).abs() < 1e-4));
    }

    #[test]
    fn envelope_keeps_extremes() {
        let mut pts: Vec<(f32, f32)> = (0..1000).map(|i| (i as f32, 1.0)).collect();
        pts[500].1 = 9.0;
        let e = envelope(&pts, 0.0, 999.0, 50);
        assert!(e.len() <= 100);
        assert!(e.iter().any(|p| p.1 == 9.0));
    }

    #[test]
    fn memo_follows_the_allocation() {
        let a = Arc::new(vec![(0.0, 1.0), (1.0, 3.0)]);
        let s1 = smoothed(&a, 0.5);
        assert!(Arc::ptr_eq(&s1, &smoothed(&a, 0.5)));
        let b = Arc::new(vec![(0.0, 1.0), (1.0, 3.0)]);
        assert!(!Arc::ptr_eq(&s1, &smoothed(&b, 0.5)));
        assert!(!Arc::ptr_eq(&s1, &smoothed(&a, 0.9)));
    }

    #[test]
    fn nearest_point() {
        let pts = vec![(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)];
        assert_eq!(nearest(&pts, 1.4), Some(1));
        assert_eq!(nearest(&pts, 1.6), Some(2));
        assert_eq!(nearest(&pts, -3.0), Some(0));
        assert_eq!(nearest(&[], 1.0), None);
    }
}
