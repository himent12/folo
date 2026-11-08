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

use crate::theme::{SANS, Theme, alpha, ty};

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
    /// Dotted (wins over dashed).
    pub dotted: bool,
    /// Any other dash pattern (wins over both), for sets of more than three lines.
    pub pattern: Option<&'static [f32]>,
    /// Draw a live dot with a halo on the last point.
    pub live: bool,
}

impl Series {
    pub fn dashed(mut self) -> Self {
        self.dashed = true;
        self
    }

    pub fn dotted(mut self) -> Self {
        self.dotted = true;
        self
    }

    pub fn pattern(mut self, dash: &'static [f32]) -> Self {
        self.pattern = Some(dash);
        self
    }

    pub fn new(name: impl Into<SharedString>, color: Hsla, points: Arc<Vec<(f32, f32)>>) -> Self {
        Self {
            name: name.into(),
            color,
            points,
            smooth: 0.0,
            area: false,
            width: 1.5,
            dashed: false,
            dotted: false,
            pattern: None,
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
    /// A fixed label lane, so the same marker sits at the same height in sibling charts.
    pub lane: Option<usize>,
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
    /// Axis titles (x, y): y above the plot's top-left, x centred under the tick labels.
    pub axis_titles: Option<(&'static str, &'static str)>,
    /// Marker label lanes to reserve even when fewer are used, so charts in a row share a plot top.
    pub min_lanes: usize,
    /// X is a count (epochs): ticks land on whole numbers and the last one labels the end.
    pub x_int: bool,
    /// Draw the floating readout box on hover. Off when the panel's legend shows the values.
    pub readout: bool,
    /// Fixed x tick count (shared across linked charts); None picks one from the width.
    pub x_ticks: Option<usize>,
    /// What the data belongs to (`ease_key(run or model id)`): easing never blends two owners.
    pub ease_key: u64,
    /// Print a shared power of ten above the y axis. Off when the panel caption carries it.
    pub pow_caption: bool,
    /// The part of x not reached yet (a live run's remaining epochs): tinted, with a label.
    pub future: Option<(f32, SharedString)>,
    /// Points to mark with a dot (x, y, colour), e.g. the best epoch.
    pub dots: Vec<(f32, f32, Hsla)>,
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
            axis_titles: None,
            min_lanes: 0,
            x_int: false,
            readout: true,
            x_ticks: None,
            future: None,
            ease_key: 0,
            pow_caption: true,
            dots: Vec::new(),
        }
    }
}

/// Height of one marker-label lane above a plot.
const LANE: f32 = 16.;

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
/// X-axis ticks that end exactly on the range when a round step divides it (0–120 → every 20),
/// otherwise the usual nice ticks.
pub fn even_ticks(lo: f32, hi: f32, max_n: usize) -> Vec<f32> {
    let range = hi - lo;
    if range > 0.0 && range.is_finite() {
        let mag = 10f32.powf((range / max_n as f32).log10().floor());
        for k in [1.0, 10.0, 100.0] {
            for m in [1.0, 2.0, 2.5, 5.0] {
                let step = m * mag * k;
                let n = range / step;
                if n <= max_n as f32
                    && n >= 2.0
                    && (n - n.round()).abs() < 1e-3
                    && (lo / step - (lo / step).round()).abs() < 1e-3
                {
                    return (0..=n.round() as usize).map(|i| lo + step * i as f32).collect();
                }
            }
        }
    }
    ticks(lo, hi, max_n)
}

/// Whole-number ticks for count axes: a 1-2-5 step that fits `max_n` labels, always ending on `hi`
/// (0–3 → 0 1 2 3; 0–120 → every 20; 0–7 → 0 5 7).
pub fn int_ticks(lo: f32, hi: f32, max_n: usize) -> Vec<f32> {
    let (lo, hi) = (lo.round(), hi.round().max(lo.round() + 1.0));
    let even = even_ticks(lo, hi, max_n);
    if even.iter().all(|v| v.fract() == 0.0) && even.last() == Some(&hi) {
        return even;
    }
    let range = hi - lo;
    // A step that divides the range exactly keeps every gap equal (0–120 → 0 40 80 120).
    for mag in [1.0f32, 10.0, 100.0, 1000.0, 10000.0] {
        for m in [1.0f32, 2.0, 3.0, 4.0, 5.0] {
            let step = m * mag;
            let n = range / step;
            if n.fract() == 0.0 && n >= 2.0 && n <= max_n.max(2) as f32 && (lo / step).fract() == 0.0 {
                return (0..=n as usize).map(|i| lo + step * i as f32).collect();
            }
        }
    }
    let mut step = 1.0f32;
    'find: for mag in [1.0f32, 10.0, 100.0, 1000.0, 10000.0] {
        for m in [1.0f32, 2.0, 5.0] {
            step = m * mag;
            if range / step <= max_n.max(1) as f32 {
                break 'find;
            }
        }
    }
    let mut out: Vec<f32> = Vec::new();
    let mut v = (lo / step).ceil() * step;
    while v < hi - 1e-3 {
        out.push(v);
        v += step;
    }
    // End on the range; drop the previous tick when the end would crowd it.
    if let Some(&last) = out.last()
        && hi - last < step * 0.6
        && out.len() > 1
    {
        out.pop();
    }
    out.push(hi);
    out
}

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

/// The stride (in octaves) of a log axis over `y0..y1` with room for `max` labels.
fn pow2_stride(y0: f32, y1: f32, max: usize) -> i32 {
    let e0 = y0.max(1e-6).log2().ceil() as i32;
    let e1 = (y1.max(1e-6) * 1.0001).log2().floor() as i32;
    let n = (e1 - e0 + 1).max(1) as usize;
    n.div_ceil(max.max(1)) as i32
}

/// Powers of two from `y0` to `y1` (both > 0), for log axes, at most about `max` of them.
#[cfg(test)]
pub fn pow2_ticks(y0: f32, y1: f32, max: usize) -> Vec<f32> {
    pow2_ticks_every(y0, y1, pow2_stride(y0, y1, max))
}

/// [`pow2_ticks`] at a given stride of `k` octaves.
pub fn pow2_ticks_every(y0: f32, y1: f32, k: i32) -> Vec<f32> {
    let e0 = y0.max(1e-6).log2().ceil() as i32;
    let e1 = (y1.max(1e-6) * 1.0001).log2().floor() as i32;
    if e1 < e0 {
        return Vec::new();
    }
    // More powers than fit: every k-th, counted down from the top (8 2 0.5), so the top power
    // keeps its label and every step is the same factor.
    let k = k.max(1);
    (e0..=e1).filter(|e| (e1 - e) % k == 0).map(|e| 2f32.powi(e)).collect()
}

/// The power of ten a y axis over `y0..y1` with `ticks` ticks shows once, if its values are tiny
/// (tick step under 0.01). Legends of that chart format their values in the same power.
pub fn axis_pow(y0: f32, y1: f32, ticks: usize) -> Option<i32> {
    let step = nice_step(y1 - y0, ticks);
    (step > 0. && step < 0.01).then(|| step.log10().floor() as i32)
}

/// An exponent in superscript digits ("-3" → "⁻³").
pub fn superscript(p: i32) -> String {
    p.to_string()
        .chars()
        .map(|c| match c {
            '-' => '⁻',
            '0' => '⁰',
            '1' => '¹',
            '2' => '²',
            '3' => '³',
            '4' => '⁴',
            '5' => '⁵',
            '6' => '⁶',
            '7' => '⁷',
            '8' => '⁸',
            _ => '⁹',
        })
        .collect()
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

/// How long live points take to move into place (DIRECTION: motion).
const EASE_MS: f32 = 240.;

/// Live series longer than this jump instead of easing: a new point among thousands moves no
/// pixel, and re-deriving a long history every frame isn't worth it.
const EASE_MAX_POINTS: usize = 512;

/// A series's points, shared between renders.
pub type Points = Arc<Vec<(f32, f32)>>;
type Pts = Points;

/// A chart's last data and the move from the previous data to it.
struct Ease {
    /// What the data belongs to (run or model, and the series names): a change means new data.
    key: (u64, Vec<SharedString>),
    /// Per series: the points it moves from, or None when it isn't moving.
    from: Vec<Option<Pts>>,
    to: Vec<Pts>,
    start: std::time::Instant,
}

thread_local! {
    static EASES: std::cell::RefCell<std::collections::HashMap<SharedString, Ease>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
}

/// Points between two revisions of a series at progress `e`: shared points move vertically, new
/// points slide out from the old last point.
pub fn tween(from: &[(f32, f32)], to: &[(f32, f32)], e: f32) -> Vec<(f32, f32)> {
    let lerp = |a: f32, b: f32| a + (b - a) * e;
    to.iter()
        .enumerate()
        .map(|(i, &(x, y))| match from.get(i) {
            Some(&(_, fy)) => (x, lerp(fy, y)),
            None => from.last().map_or((x, y), |&(lx, ly)| (lerp(lx, x), lerp(ly, y))),
        })
        .collect()
}

/// An update that only appends or revises the tail of the same series (a live run), as opposed
/// to different data (another run, a toggled series), which should jump.
fn is_append(from: &[(f32, f32)], to: &[(f32, f32)]) -> bool {
    !from.is_empty()
        && to.len() >= from.len()
        && to.len() - from.len() <= 8.max(from.len() / 20)
        && to[0].0 == from[0].0
        && to[from.len() - 1].0 == from[from.len() - 1].0
}

/// Same points, bit for bit (a rebuilt allocation of unchanged data counts as the same).
fn same_points(a: &Pts, b: &Pts) -> bool {
    Arc::ptr_eq(a, b)
        || (a.len() == b.len()
            && a.iter()
                .zip(b.iter())
                .all(|(p, q)| p.0.to_bits() == q.0.to_bits() && p.1.to_bits() == q.1.to_bits()))
}

/// A stable key for what a chart's data belongs to (a run or model id).
pub fn ease_key(id: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    id.hash(&mut h);
    h.finish()
}

/// One frame of easing for chart `id`: the points to draw instead of `to` while a move runs.
fn ease_frame(
    id: &SharedString,
    key: (u64, Vec<SharedString>),
    to: &[Pts],
    now: std::time::Instant,
    reduce: bool,
) -> Option<Vec<Pts>> {
    let progress = |start: std::time::Instant| (now - start).as_secs_f32() * 1000. / EASE_MS;
    let settled = now - std::time::Duration::from_secs(1);
    EASES.with(|m| {
        let mut m = m.borrow_mut();
        let e = m.entry(id.clone()).or_insert_with(|| Ease {
            key: key.clone(),
            from: vec![None; to.len()],
            to: to.to_vec(),
            start: settled,
        });
        if reduce {
            // Reduce motion: no move starts, and one in progress stops where its data is.
            *e = Ease { key, from: vec![None; to.len()], to: to.to_vec(), start: settled };
            return None;
        }
        let unchanged =
            e.key == key && e.to.len() == to.len() && e.to.iter().zip(to).all(|(a, b)| same_points(a, b));
        if unchanged {
            // Same data, perhaps rebuilt: adopt the new allocations; a move in progress goes on.
            e.to = to.to_vec();
        } else {
            let p = progress(e.start).min(1.);
            let related = e.key == key && e.to.len() == to.len() && e.from.len() == to.len();
            // One slot per series, always: a hidden-then-restored series can ease again.
            let from: Vec<Option<Pts>> = if !related {
                vec![None; to.len()]
            } else {
                e.from
                    .iter()
                    .zip(&e.to)
                    .zip(to)
                    .map(|((f, old), new)| {
                        if same_points(old, new) {
                            return None;
                        }
                        // Start from what is on screen now (mid-move if this series was moving).
                        let cur = match f {
                            Some(f) if p < 1. => tween(f, old, ease_out(p)),
                            _ => old.to_vec(),
                        };
                        (new.len() <= EASE_MAX_POINTS && is_append(&cur, new)).then(|| Arc::new(cur))
                    })
                    .collect()
            };
            let moving = from.iter().any(Option::is_some);
            *e = Ease { key, from, to: to.to_vec(), start: if moving { now } else { settled } };
        }
        let p = progress(e.start);
        if p >= 1. || e.from.iter().all(Option::is_none) {
            return None;
        }
        let k = ease_out(p);
        // Series that aren't moving keep their allocation, so their cached derivations hold.
        Some(
            e.from
                .iter()
                .zip(&e.to)
                .map(|(f, t)| f.as_ref().map_or(t.clone(), |f| Arc::new(tween(f, t, k))))
                .collect(),
        )
    })
}

/// Ease new live data in over 240 ms (instant under Reduce motion). Numbers elsewhere are not
/// animated; only this chart's geometry is.
fn ease(spec: &mut LineChart, window: &mut Window) {
    let key = (spec.ease_key, spec.series.iter().map(|s| s.name.clone()).collect());
    let to: Vec<Pts> = spec.series.iter().map(|s| s.points.clone()).collect();
    if let Some(pts) = ease_frame(&spec.id, key, &to, std::time::Instant::now(), crate::ui::reduce_motion()) {
        for (s, p) in spec.series.iter_mut().zip(pts) {
            s.points = p;
        }
        window.request_animation_frame();
    }
}

fn ease_out(p: f32) -> f32 {
    1. - (1. - p.clamp(0., 1.)).powi(3)
}

/// The y at `x` if the series has a point within 6 % of `span` of it (the cursor-dot rule), else
/// None: past a curve's end there is no value to show.
pub fn value_near(points: &[(f32, f32)], x: f32, span: f32) -> Option<f32> {
    let (px_, py) = points[nearest(points, x)?];
    ((px_ - x).abs() <= span * 0.06).then_some(py)
}

/// The cursor chip's text, named ("epoch 36") so a linked cursor's unit is explicit.
fn chip_label(spec: &LineChart, x: f32) -> String {
    let v = (spec.x_fmt)(x);
    if spec.x_name.is_empty() { v } else { format!("{} {v}", spec.x_name) }
}

/// Index of the point nearest to `x` in a sorted series.
/// The value a legend shows: at the shared cursor when there is one, else the latest point.
pub fn value_at(points: &[(f32, f32)], hover: Option<f32>, x0: f32, x1: f32) -> Option<f32> {
    match hover {
        Some(f) => nearest(points, x0 + (x1 - x0) * f).map(|i| points[i].1),
        None => points.last().map(|p| p.1),
    }
}

/// The x the cursor snaps to at `hx`: the first series point within 6 % of the range (the same
/// rule the cursor painter uses), or None when the cursor is past the data.
pub fn cursor_x(series: &[Series], hx: f32, span: f32) -> Option<f32> {
    series.iter().find_map(|s| {
        let data = if s.smooth > 0.0 { smoothed(&s.points, s.smooth) } else { s.points.clone() };
        let (x, _) = data[nearest(&data, hx)?];
        ((x - hx).abs() <= span * 0.06 || s.points.len() <= 1).then_some(x)
    })
}

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
            ..gpui::font(SANS)
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

pub fn text_width(window: &mut Window, text: &str, size_px: f32) -> Pixels {
    let run = TextRun {
        len: text.len(),
        font: gpui::Font {
            features: gpui::FontFeatures(Arc::new(vec![("tnum".into(), 1)])),
            ..gpui::font(SANS)
        },
        color: gpui::black(),
        background_color: None,
        underline: None,
        strikethrough: None,
    };
    window.text_system().shape_line(SharedString::from(text.to_string()), px(size_px), &[run], None).width
}

/// Dash patterns (on, off) in px.
const DASHED: &[f32] = &[4., 3.];
const DOTTED: &[f32] = &[1.5, 3.];
/// Long dashes and dash-dot, for line sets beyond solid / dashed / dotted (the PR curves).
pub const LONG_DASH: &[f32] = &[9., 3.];
pub const DASH_DOT: &[f32] = &[6., 2.5, 1.5, 2.5];

fn stroke_path(window: &mut Window, pts: &[Point<Pixels>], width: f32, color: Hsla, dash: &[f32]) {
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
    let label_size = ty::XS.size - 0.5;
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
    let yticks = if spec.log_y { pow2_ticks_every(y0, y1, stride) } else { ticks(y0, y1, spec.y_ticks) };
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
        let w = text_width(window, &m.label, label_size) + px(8.);
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
        paint_text(window, cx, yt, point(bounds.left(), bounds.top()), t.text_tertiary, ty::XS.size, false);
        let w = text_width(window, xt, ty::XS.size);
        paint_text(
            window,
            cx,
            xt,
            point(plot.left() + (plot.size.width - w) / 2., plot.bottom() + px(22.)),
            t.text_tertiary,
            ty::XS.size,
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
            let w = text_width(window, label, label_size);
            // At the start of the span (where "now" is): in the marker band, like the other
            // annotations, when the chart has one and lane 0 is free there; else inside the plot.
            let lx = (fx + px(6.)).min(plot.right() - w - px(6.));
            let in_band = band > 0.
                && !marker_layout
                    .iter()
                    .any(|(ox, ow, lane, _)| *lane == 0 && *ox < lx + w + px(6.) && lx < *ox + *ow + px(6.));
            let y = if in_band { plot.top() - px(band) } else { plot.top() + px(4.) };
            paint_text(window, cx, label, point(lx, y), t.text_tertiary, label_size, false);
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
            let left = (sx(v) - w / 2.).clamp(plot.left() - px(4.), plot.right() - w);
            (left, left + w)
        });
    for v in &xt {
        let x = sx(*v);
        window.paint_quad(fill(Bounds::new(point(x, plot.bottom()), size(px(1.), px(4.))), t.chart_axis));
        let l = (spec.x_fmt)(*v);
        let w = text_width(window, &l, label_size);
        // Centred on its tick (the end labels too: "0.00" / "1.00" flush with the tick read as
        // belonging beside it); only the chart's own edges push a label inward.
        let lx = (x - w / 2.).clamp(bounds.left(), bounds.right() - w);
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
        Some(gpui::ContentMask {
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
            label_size,
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
        let w = text_width(window, &l, label_size) + px(10.);
        let chip = Bounds::new(
            point((cx_px - w / 2.).clamp(plot.left() - px(4.), plot.right() - w), plot.bottom() + px(3.)),
            size(w, px(16.)),
        );
        window.paint_quad(fill(chip, t.text_primary).corner_radii(px(3.)));
        paint_text(
            window,
            cx,
            &l,
            point(chip.left() + px(5.), chip.top() + px(1.)),
            t.bg_panel,
            label_size,
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
    let fs = ty::XS.size;
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
    sparkline_in(t, values, color, height, live, None)
}

/// A sparkline over a fixed value range (e.g. 0..1 for utilisation), so small wobbles stay small.
pub fn sparkline_in(
    t: &Theme,
    values: Arc<Vec<f32>>,
    color: Hsla,
    height: f32,
    live: bool,
    range: Option<(f32, f32)>,
) -> AnyElement {
    let t = *t;
    canvas(
        |_, _, _| (),
        move |b, _, window, _| {
            if values.len() < 2 {
                return;
            }
            let (lo, hi) = range.unwrap_or_else(|| {
                values.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &v| (lo.min(v), hi.max(v)))
            });
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
            stroke_path(window, &pts, 1.25, color, &[]);
            if live {
                let c = pts[n - 1];
                window.paint_quad(
                    fill(Bounds::centered_at(c, size(px(4.), px(4.))), color).corner_radii(px(2.)),
                );
            }
        },
    )
    .w_full()
    .map(|d| if height > 0. { d.h(px(height)) } else { d.h_full() })
    .into_any_element()
}

/// Vertical bars for a histogram; `labels` sit under the bars (one per bin).
pub fn histogram(
    t: &Theme,
    id: impl Into<SharedString>,
    counts: Arc<Vec<u64>>,
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
                            let s = crate::ui::int(c);
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
        assert_eq!(even_ticks(0.0, 120.0, 8), vec![0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0]);
        assert_eq!(int_ticks(0.0, 3.0, 3), vec![0.0, 1.0, 2.0, 3.0]);
        assert_eq!(int_ticks(0.0, 120.0, 8), vec![0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0]);
        assert_eq!(int_ticks(0.0, 7.0, 3), vec![0.0, 7.0]);
        assert_eq!(int_ticks(0.0, 120.0, 3), vec![0.0, 40.0, 80.0, 120.0]);
        assert_eq!(int_ticks(0.0, 100.0, 3), vec![0.0, 50.0, 100.0]);
        assert_eq!(int_ticks(0.0, 103.0, 6), vec![0.0, 20.0, 40.0, 60.0, 80.0, 103.0]);
        assert!(int_ticks(0.0, 5.0, 2).iter().all(|v| v.fract() == 0.0));
        assert_eq!(even_ticks(0.0, 100.0, 5), vec![0.0, 20.0, 40.0, 60.0, 80.0, 100.0]);
        assert_eq!(even_ticks(0.0, 5.0, 8), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(even_ticks(0.0, 1.0, 5), vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0]);
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
    fn cursor_past_the_data_has_no_chip() {
        let pts: Vec<(f32, f32)> = (0..=40).map(|i| (i as f32, 0.5)).collect();
        let s = vec![Series::new("a", gpui::hsla(0., 0., 0., 1.), Arc::new(pts))];
        // Epoch 100 of 120 with data through 40: no snap, so no tick is hidden.
        assert_eq!(cursor_x(&s, 100., 120.), None);
        assert_eq!(cursor_x(&s, 39.6, 120.), Some(40.));
    }

    #[test]
    fn tween_moves_shared_points_and_slides_new_ones() {
        let from = [(0.0, 0.0), (1.0, 1.0)];
        let to = [(0.0, 0.0), (1.0, 3.0), (2.0, 5.0)];
        assert_eq!(tween(&from, &to, 0.0), vec![(0.0, 0.0), (1.0, 1.0), (1.0, 1.0)]);
        assert_eq!(tween(&from, &to, 0.5), vec![(0.0, 0.0), (1.0, 2.0), (1.5, 3.0)]);
        assert_eq!(tween(&from, &to, 1.0), to.to_vec());
        // Appending to the same series eases; different data jumps.
        assert!(is_append(&from, &to));
        assert!(!is_append(&from, &[(5.0, 0.0), (6.0, 1.0)]));
        assert!(!is_append(&from, &[(0.0, 0.0)]));
    }

    #[test]
    fn rebuilt_identical_data_never_restarts_easing() {
        let id: SharedString = "test-rebuilt".into();
        let key = || (1u64, vec![SharedString::from("a")]);
        let pts = |n: usize| Arc::new((0..n).map(|i| (i as f32, i as f32)).collect::<Vec<_>>());
        let t0 = std::time::Instant::now();
        assert!(ease_frame(&id, key(), &[pts(10)], t0, false).is_none());
        // Every render rebuilds the same points (like the PR chart): still nothing to animate.
        for _ in 0..3 {
            assert!(ease_frame(&id, key(), &[pts(10)], t0, false).is_none());
        }
        // An append eases, and the move ends.
        assert!(ease_frame(&id, key(), &[pts(11)], t0, false).is_some());
        let later = t0 + std::time::Duration::from_millis(300);
        assert!(ease_frame(&id, key(), &[pts(11)], later, false).is_none());
        // Another owner (a different run) with an "append-like" shape jumps.
        assert!(ease_frame(&id, (2, vec!["a".into()]), &[pts(12)], later, false).is_none());
        // Long histories jump too.
        let long = "test-long".into();
        assert!(ease_frame(&long, key(), &[pts(600)], t0, false).is_none());
        assert!(ease_frame(&long, key(), &[pts(601)], t0, false).is_none());
    }

    #[test]
    fn easing_survives_hidden_series_and_stops_under_reduce_motion() {
        let id: SharedString = "test-hide".into();
        let names =
            |n: usize| (7u64, (0..n).map(|i| SharedString::from(format!("s{i}"))).collect::<Vec<_>>());
        let pts = |n: usize| Arc::new((0..n).map(|i| (i as f32, i as f32)).collect::<Vec<_>>());
        let t0 = std::time::Instant::now();
        assert!(ease_frame(&id, names(2), &[pts(10), pts(10)], t0, false).is_none());
        // Hide one series, then restore both: appends ease again.
        assert!(ease_frame(&id, names(1), &[pts(10)], t0, false).is_none());
        assert!(ease_frame(&id, names(2), &[pts(10), pts(10)], t0, false).is_none());
        assert!(ease_frame(&id, names(2), &[pts(11), pts(11)], t0, false).is_some());
        // Reduce motion mid-move: the move stops at once.
        let mid = t0 + std::time::Duration::from_millis(100);
        assert!(ease_frame(&id, names(2), &[pts(11), pts(11)], mid, true).is_none());
        assert!(ease_frame(&id, names(2), &[pts(11), pts(11)], mid, false).is_none());
    }

    #[test]
    fn no_value_past_a_curves_end() {
        let pts = [(0.0, 1.0), (0.2, 0.9), (0.4, 0.5)];
        assert_eq!(value_near(&pts, 0.41, 1.), Some(0.5));
        assert_eq!(value_near(&pts, 0.9, 1.), None);
    }

    #[test]
    fn log_axes_tick_at_powers_of_two() {
        assert_eq!(pow2_ticks(0.4, 9., 8), vec![0.5, 1., 2., 4., 8.]);
        assert_eq!(pow2_ticks(1., 1.5, 8), vec![1.]);
        // 0.001..10 is 13 powers; in room for 4, every fourth, down from 8.
        assert_eq!(pow2_ticks(0.001, 10., 4), vec![2f32.powi(-9), 2f32.powi(-5), 0.5, 8.]);
        // 0.25..8 in room for 3: every other power down from the top, even steps, domain kept.
        assert_eq!(pow2_stride(0.25, 8., 3), 2);
        assert_eq!(pow2_ticks(0.25, 8., 3), vec![0.5, 2., 8.]);
        // 0.001..2 is 11 powers: every third, down from 2.
        assert_eq!(pow2_ticks(0.001, 2., 4), vec![2f32.powi(-8), 2f32.powi(-5), 0.25, 2.]);
    }

    #[test]
    fn small_values_share_a_power_of_ten() {
        assert_eq!(superscript(-3), "⁻³");
        assert_eq!(axis_pow(0., 0.012, 3), Some(-3));
        assert_eq!(axis_pow(0., 1., 4), None);
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
