// Adapted from GPUI (Apache-2.0): crates/gpui/src/path_builder.rs
//! `PathBuilder`: GPUI's path API (lyon's SVG-style builder: `move_to`, `line_to`, curves, `close`),
//! resolved at `build` into flattened contours in logical pixels with a fill rule or a stroke style.
//! GPUI tessellates the same path with lyon; here the renderer computes coverage from the contours, so
//! what must match is the geometry: lyon's defaults (fill: even-odd; stroke: butt caps, miter joins,
//! miter limit 4; tolerance 0.1) and GPUI's dashing (dashes cut along the whole path's length, each
//! "on" dash its own open contour).

use crate::{
    Background, Bounds, Contour, FillRule, LineCap, LineJoin, PathData, PathKind, Pixels, Point,
    ScaledPixels, Scene, StrokeStyle, StylePaint, point, px,
};
use std::fmt::Debug;
use std::sync::Arc;

/// How a stroked path is drawn (lyon's `StrokeOptions` as GPUI uses them).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StrokeOptions {
    /// The line width in logical pixels.
    pub line_width: f32,
    /// The cap at open ends.
    pub line_cap: LineCap,
    /// The join between segments.
    pub line_join: LineJoin,
    /// The miter limit (beyond it a miter join is beveled).
    pub miter_limit: f32,
    /// The maximum distance between a curve and its flattened polyline.
    pub tolerance: f32,
}

impl Default for StrokeOptions {
    fn default() -> Self {
        Self {
            line_width: 1.,
            line_cap: LineCap::Butt,
            line_join: LineJoin::Miter,
            miter_limit: 4.,
            tolerance: 0.1,
        }
    }
}

impl StrokeOptions {
    /// These options with `width`.
    pub fn with_line_width(mut self, width: f32) -> Self {
        self.line_width = width;
        self
    }
}

/// How a filled path is drawn (lyon's `FillOptions` defaults: even-odd).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FillOptions {
    /// Which points are inside.
    pub fill_rule: FillRule,
    /// The maximum distance between a curve and its flattened polyline.
    pub tolerance: f32,
}

impl Default for FillOptions {
    fn default() -> Self {
        Self { fill_rule: FillRule::EvenOdd, tolerance: 0.1 }
    }
}

/// Whether a [`PathBuilder`] strokes or fills.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PathStyle {
    /// Stroked.
    Stroke(StrokeOptions),
    /// Filled.
    Fill(FillOptions),
}

/// One contour of a built path.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PathContour<P: Clone + Debug + Default + PartialEq> {
    /// The points, in order (curves already flattened).
    pub points: Vec<Point<P>>,
    /// Whether the last point joins the first.
    pub closed: bool,
}

/// How a built path is painted.
#[derive(Clone, Debug, PartialEq)]
pub enum PathMode<P> {
    /// Filled with a rule.
    Fill(FillRule),
    /// Stroked.
    Stroke {
        /// The line width.
        width: P,
        /// The cap at open ends.
        cap: LineCap,
        /// The join between segments.
        join: LineJoin,
        /// The miter limit.
        miter_limit: f32,
    },
}

/// A built path (GPUI's `Path<Pixels>`): contours, how to paint them, and the painted area's bounds.
#[derive(Clone, Debug, PartialEq)]
pub struct Path<P: Clone + Debug + Default + PartialEq> {
    /// The contours (dashes already cut into their own contours).
    pub contours: Vec<PathContour<P>>,
    /// Fill or stroke.
    pub mode: PathMode<P>,
    /// What the path covers (for a stroke, the outline including caps and miters).
    pub bounds: Bounds<P>,
}

impl Path<Pixels> {
    /// This path in device pixels.
    pub fn scale(&self, factor: f32) -> Path<ScaledPixels> {
        let contours = self
            .contours
            .iter()
            .map(|c| PathContour {
                points: c.points.iter().map(|p| p.scale(factor)).collect(),
                closed: c.closed,
            })
            .collect();
        let mode = match &self.mode {
            PathMode::Fill(rule) => PathMode::Fill(*rule),
            PathMode::Stroke { width, cap, join, miter_limit } => PathMode::Stroke {
                width: width.scale(factor),
                cap: *cap,
                join: *join,
                miter_limit: *miter_limit,
            },
        };
        Path { contours, mode, bounds: self.bounds.scale(factor) }
    }
}

impl Path<ScaledPixels> {
    /// The scene's geometry for this path (a new id).
    pub fn to_path_data(&self) -> PathData {
        let contours =
            self.contours.iter().map(|c| Contour { points: c.points.clone(), closed: c.closed }).collect();
        let kind = match &self.mode {
            PathMode::Fill(rule) => PathKind::Fill(*rule),
            PathMode::Stroke { width, cap, join, miter_limit } => PathKind::Stroke(StrokeStyle {
                width: *width,
                cap: *cap,
                join: *join,
                miter_limit: *miter_limit,
                dash: None,
            }),
        };
        PathData::new(contours, kind)
    }
}

/// Pushes a path as GPUI's `Window::paint_path` does: scaled to device pixels, the paint faded by the
/// element opacity, clipped by the content mask, skipped when its bounds are outside the mask.
pub fn paint_path(scene: &mut Scene, cx: &StylePaint, path: &Path<Pixels>, color: impl Into<Background>) {
    let path = path.scale(cx.scale_factor);
    let clip = cx.content_mask.scale(cx.scale_factor).bounds;
    if path.bounds.intersect(&clip).is_empty() {
        return;
    }
    let paint: Background = color.into();
    scene.push_path(crate::scene::Path {
        bounds: path.bounds,
        clip,
        path: Arc::new(path.to_path_data()),
        paint: paint.opacity(cx.opacity),
    });
}

#[derive(Clone, Copy, Debug)]
enum Segment {
    Line(Point<Pixels>),
    Quadratic { ctrl: Point<Pixels>, to: Point<Pixels> },
    Cubic { ctrl1: Point<Pixels>, ctrl2: Point<Pixels>, to: Point<Pixels> },
}

#[derive(Clone, Debug)]
struct SubPath {
    start: Point<Pixels>,
    segments: Vec<Segment>,
    closed: bool,
}

/// An affine transform `x' = a x + c y + e`, `y' = b x + d y + f` (lyon's `Transform` layout).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Affine([f32; 6]);

impl Affine {
    fn then(self, next: Affine) -> Affine {
        let [a, b, c, d, e, f] = self.0;
        let [na, nb, nc, nd, ne, nf] = next.0;
        Affine([
            a * na + b * nc,
            a * nb + b * nd,
            c * na + d * nc,
            c * nb + d * nd,
            e * na + f * nc + ne,
            e * nb + f * nd + nf,
        ])
    }

    fn apply(&self, p: Point<Pixels>) -> Point<Pixels> {
        let [a, b, c, d, e, f] = self.0;
        let (x, y) = (f32::from(p.x), f32::from(p.y));
        point(px(a * x + c * y + e), px(b * x + d * y + f))
    }
}

/// Builds a filled or stroked [`Path`] (GPUI's `PathBuilder`).
#[derive(Clone, Debug)]
pub struct PathBuilder {
    subpaths: Vec<SubPath>,
    in_subpath: bool,
    current: Point<Pixels>,
    first: Point<Pixels>,
    transform: Option<Affine>,
    /// Fill or stroke, and their options.
    pub style: PathStyle,
    dash_array: Option<Vec<Pixels>>,
}

impl Default for PathBuilder {
    fn default() -> Self {
        Self {
            subpaths: Vec::new(),
            in_subpath: false,
            current: Point::default(),
            first: Point::default(),
            transform: None,
            style: PathStyle::Fill(FillOptions::default()),
            dash_array: None,
        }
    }
}

impl PathBuilder {
    /// A builder for a stroke `width` wide.
    pub fn stroke(width: Pixels) -> Self {
        Self {
            style: PathStyle::Stroke(StrokeOptions::default().with_line_width(width.into())),
            ..Self::default()
        }
    }

    /// A builder for a fill.
    pub fn fill() -> Self {
        Self::default()
    }

    /// Sets fill or stroke and their options.
    pub fn with_style(self, style: PathStyle) -> Self {
        Self { style, ..self }
    }

    /// Sets the dash lengths (alternating on, off; an odd list is repeated to make it even, as SVG does).
    pub fn dash_array(mut self, dash_array: &[Pixels]) -> Self {
        let mut array = dash_array.to_vec();
        if dash_array.len() % 2 == 1 {
            array.extend_from_slice(dash_array);
        }
        self.dash_array = Some(array);
        self
    }

    /// Starts a new contour at `to` (ending the current one, open).
    pub fn move_to(&mut self, to: Point<Pixels>) {
        self.subpaths.push(SubPath { start: to, segments: Vec::new(), closed: false });
        self.in_subpath = true;
        self.current = to;
        self.first = to;
    }

    fn push(&mut self, segment: Segment, to: Point<Pixels>) {
        if !self.in_subpath {
            self.move_to(self.current);
        }
        if let Some(subpath) = self.subpaths.last_mut() {
            subpath.segments.push(segment);
        }
        self.current = to;
    }

    /// A straight line from the current point to `to`.
    pub fn line_to(&mut self, to: Point<Pixels>) {
        self.push(Segment::Line(to), to);
    }

    /// A quadratic curve to `to` with control point `ctrl`.
    pub fn curve_to(&mut self, to: Point<Pixels>, ctrl: Point<Pixels>) {
        self.push(Segment::Quadratic { ctrl, to }, to);
    }

    /// A cubic Bézier to `to` with control points `control_a` and `control_b`.
    pub fn cubic_bezier_to(&mut self, to: Point<Pixels>, control_a: Point<Pixels>, control_b: Point<Pixels>) {
        self.push(Segment::Cubic { ctrl1: control_a, ctrl2: control_b, to }, to);
    }

    /// Adds a polygon as its own contour.
    pub fn add_polygon(&mut self, points: &[Point<Pixels>], closed: bool) {
        let Some((&first, rest)) = points.split_first() else {
            return;
        };
        self.move_to(first);
        for &p in rest {
            self.line_to(p);
        }
        if closed {
            self.close();
        } else {
            self.in_subpath = false;
        }
    }

    /// Closes the current contour; the current point returns to its start.
    pub fn close(&mut self) {
        if self.in_subpath {
            if let Some(subpath) = self.subpaths.last_mut() {
                subpath.closed = true;
            }
            self.in_subpath = false;
            self.current = self.first;
        }
    }

    fn then_transform(&mut self, next: Affine) {
        self.transform = Some(self.transform.map_or(next, |t| t.then(next)));
    }

    /// Translates the whole path by `to` (after any transform set so far).
    pub fn translate(&mut self, to: Point<Pixels>) {
        self.then_transform(Affine([1., 0., 0., 1., to.x.into(), to.y.into()]));
    }

    /// Scales the whole path by `scale` (after any transform set so far).
    pub fn scale(&mut self, scale: f32) {
        self.then_transform(Affine([scale, 0., 0., scale, 0., 0.]));
    }

    /// Rotates the whole path by `angle` degrees (after any transform set so far).
    pub fn rotate(&mut self, angle: f32) {
        let (sin, cos) = angle.to_radians().sin_cos();
        self.then_transform(Affine([cos, sin, -sin, cos, 0., 0.]));
    }

    /// Resolves the path: transform, flatten, dash, and compute the painted bounds.
    pub fn build(self) -> anyhow::Result<Path<Pixels>> {
        let transform = self.transform;
        let map = |p: Point<Pixels>| transform.map_or(p, |t| t.apply(p));
        let tolerance = match self.style {
            PathStyle::Stroke(options) => options.tolerance,
            PathStyle::Fill(options) => options.tolerance,
        };
        let mut contours: Vec<PathContour<Pixels>> = self
            .subpaths
            .iter()
            .filter(|s| !s.segments.is_empty())
            .map(|s| flatten(s, map, tolerance))
            .collect();

        let path = match self.style {
            PathStyle::Fill(options) => {
                let bounds = points_bounds(contours.iter().flat_map(|c| c.points.iter().copied()));
                Path { contours, mode: PathMode::Fill(options.fill_rule), bounds }
            }
            PathStyle::Stroke(options) => {
                if let Some(dashes) = &self.dash_array {
                    contours = dash(&contours, dashes);
                }
                let bounds = stroke_bounds(&contours, &options);
                let mode = PathMode::Stroke {
                    width: px(options.line_width),
                    cap: options.line_cap,
                    join: options.line_join,
                    miter_limit: options.miter_limit,
                };
                Path { contours, mode, bounds }
            }
        };
        Ok(path)
    }
}

type V = (f32, f32);

fn v(p: Point<Pixels>) -> V {
    (p.x.into(), p.y.into())
}

fn to_point(p: V) -> Point<Pixels> {
    point(px(p.0), px(p.1))
}

/// Flattens one subpath (already non-empty) into a polyline within `tolerance`.
fn flatten(
    subpath: &SubPath,
    map: impl Fn(Point<Pixels>) -> Point<Pixels>,
    tolerance: f32,
) -> PathContour<Pixels> {
    let mut points = vec![map(subpath.start)];
    let mut from = v(points[0]);
    let tolerance = tolerance.max(1e-4);
    for segment in &subpath.segments {
        match *segment {
            Segment::Line(to) => {
                let to = map(to);
                from = v(to);
                points.push(to);
            }
            Segment::Quadratic { ctrl, to } => {
                let (c, t) = (v(map(ctrl)), v(map(to)));
                let dd = len((from.0 - 2. * c.0 + t.0, from.1 - 2. * c.1 + t.1));
                let n = steps(dd / (4. * tolerance));
                for i in 1..=n {
                    let s = i as f32 / n as f32;
                    let u = 1. - s;
                    points.push(to_point((
                        u * u * from.0 + 2. * u * s * c.0 + s * s * t.0,
                        u * u * from.1 + 2. * u * s * c.1 + s * s * t.1,
                    )));
                }
                from = t;
            }
            Segment::Cubic { ctrl1, ctrl2, to } => {
                let (c1, c2, t) = (v(map(ctrl1)), v(map(ctrl2)), v(map(to)));
                let dd1 = len((from.0 - 2. * c1.0 + c2.0, from.1 - 2. * c1.1 + c2.1));
                let dd2 = len((c1.0 - 2. * c2.0 + t.0, c1.1 - 2. * c2.1 + t.1));
                let n = steps(3. * dd1.max(dd2) / (4. * tolerance));
                for i in 1..=n {
                    let s = i as f32 / n as f32;
                    let u = 1. - s;
                    let (a, b, c, d) = (u * u * u, 3. * u * u * s, 3. * u * s * s, s * s * s);
                    points.push(to_point((
                        a * from.0 + b * c1.0 + c * c2.0 + d * t.0,
                        a * from.1 + b * c1.1 + c * c2.1 + d * t.1,
                    )));
                }
                from = t;
            }
        }
    }
    PathContour { points, closed: subpath.closed }
}

/// Uniform subdivision count `ceil(sqrt(ratio))`, where `ratio` is the chord error of one segment over
/// the tolerance (the error of `n` segments falls as `1 / n²`).
fn steps(ratio: f32) -> usize {
    (ratio.sqrt().ceil() as usize).clamp(1, 1024)
}

fn len(d: V) -> f32 {
    (d.0 * d.0 + d.1 * d.1).sqrt()
}

/// GPUI's dashing: walk the whole path's length (all contours in order, a closed contour including its
/// closing segment), alternating on / off dash lengths from the start; each "on" stretch becomes an open
/// contour (split again where it crosses from one contour into the next).
fn dash(contours: &[PathContour<Pixels>], dashes: &[Pixels]) -> Vec<PathContour<Pixels>> {
    let dashes: Vec<f32> = dashes.iter().map(|d| f32::from(*d)).collect();
    // GPUI loops forever on a pattern that never advances; such a pattern strokes solid here.
    if dashes.is_empty() || dashes.iter().any(|d| *d < 0.) || dashes.iter().sum::<f32>() <= 0. {
        return contours.to_vec();
    }
    // Each contour as an open polyline with cumulative lengths, offset by the lengths before it.
    let mut lines: Vec<(Vec<V>, Vec<f32>)> = Vec::with_capacity(contours.len());
    let mut total = 0.;
    for contour in contours {
        let mut pts: Vec<V> = contour.points.iter().map(|p| v(*p)).collect();
        if contour.closed && pts.len() > 1 {
            pts.push(pts[0]);
        }
        let mut cum = Vec::with_capacity(pts.len());
        for (i, p) in pts.iter().enumerate() {
            if i > 0 {
                let q = pts[i - 1];
                total += len((p.0 - q.0, p.1 - q.1));
            }
            cum.push(total);
        }
        lines.push((pts, cum));
    }

    let mut out = Vec::new();
    let (mut pos, mut index) = (0f32, 0usize);
    while pos < total {
        let next = (pos + dashes[index % dashes.len()]).min(total);
        if index % 2 == 0 {
            for (pts, cum) in &lines {
                let (s0, s1) = (cum[0], cum[cum.len() - 1]);
                let (a, b) = (pos.max(s0), next.min(s1));
                if a < b {
                    out.push(PathContour { points: slice(pts, cum, a, b), closed: false });
                }
            }
        }
        pos = next;
        index += 1;
    }
    out
}

/// The part of a polyline between arc lengths `a < b` (both inside its range).
fn slice(pts: &[V], cum: &[f32], a: f32, b: f32) -> Vec<Point<Pixels>> {
    let at = |d: f32| -> V {
        let i = cum.partition_point(|c| *c < d).clamp(1, pts.len() - 1);
        let (l0, l1) = (cum[i - 1], cum[i]);
        let t = if l1 > l0 { (d - l0) / (l1 - l0) } else { 0. };
        let (p, q) = (pts[i - 1], pts[i]);
        (p.0 + (q.0 - p.0) * t, p.1 + (q.1 - p.1) * t)
    };
    let mut out = vec![to_point(at(a))];
    for (p, c) in pts.iter().zip(cum) {
        if *c > a && *c < b {
            out.push(to_point(*p));
        }
    }
    out.push(to_point(at(b)));
    out
}

/// The bounding box of some points (zero-sized at the origin when there are none, as GPUI's empty path).
fn points_bounds(points: impl Iterator<Item = Point<Pixels>>) -> Bounds<Pixels> {
    let mut extent: Option<(V, V)> = None;
    for p in points {
        extend(&mut extent, v(p));
    }
    extent_bounds(extent)
}

fn extend(extent: &mut Option<(V, V)>, p: V) {
    *extent = Some(match *extent {
        None => (p, p),
        Some((lo, hi)) => ((lo.0.min(p.0), lo.1.min(p.1)), (hi.0.max(p.0), hi.1.max(p.1))),
    });
}

fn extent_bounds(extent: Option<(V, V)>) -> Bounds<Pixels> {
    match extent {
        None => Bounds::default(),
        Some((lo, hi)) => Bounds::from_corners(to_point(lo), to_point(hi)),
    }
}

/// The bounds of the stroke outline: each segment's rectangle, caps, and the outer miter tips that stay
/// within the miter limit (beyond it lyon bevels, which the rectangles already cover).
fn stroke_bounds(contours: &[PathContour<Pixels>], options: &StrokeOptions) -> Bounds<Pixels> {
    let hw = options.line_width / 2.;
    let mut extent = None;
    for contour in contours {
        let mut pts: Vec<V> = contour.points.iter().map(|p| v(*p)).collect();
        pts.dedup();
        if contour.closed && pts.len() > 2 && pts.first() == pts.last() {
            pts.pop();
        }
        let n = pts.len();
        if n < 2 {
            continue;
        }
        let segs = if contour.closed { n } else { n - 1 };
        let dir = |i: usize| {
            let (p, q) = (pts[i % n], pts[(i + 1) % n]);
            let l = len((q.0 - p.0, q.1 - p.1));
            ((q.0 - p.0) / l, (q.1 - p.1) / l)
        };
        for i in 0..segs {
            let (p, q) = (pts[i], pts[(i + 1) % n]);
            let d = dir(i);
            let nrm = (-d.1 * hw, d.0 * hw);
            for c in [p, q] {
                extend(&mut extent, (c.0 + nrm.0, c.1 + nrm.1));
                extend(&mut extent, (c.0 - nrm.0, c.1 - nrm.1));
            }
        }
        if !contour.closed {
            for (end, d) in [
                (pts[0], {
                    let d = dir(0);
                    (-d.0, -d.1)
                }),
                (pts[n - 1], dir(n - 2)),
            ] {
                match options.line_cap {
                    LineCap::Butt => {}
                    LineCap::Square => {
                        let (nx, ny) = (-d.1 * hw, d.0 * hw);
                        let tip = (end.0 + d.0 * hw, end.1 + d.1 * hw);
                        extend(&mut extent, (tip.0 + nx, tip.1 + ny));
                        extend(&mut extent, (tip.0 - nx, tip.1 - ny));
                    }
                    LineCap::Round => {
                        extend(&mut extent, (end.0 - hw, end.1 - hw));
                        extend(&mut extent, (end.0 + hw, end.1 + hw));
                    }
                }
            }
        }
        let joints = if contour.closed { 0..n } else { 1..n - 1 };
        for j in joints {
            let vtx = pts[j];
            match options.line_join {
                LineJoin::Bevel => {}
                LineJoin::Round => {
                    extend(&mut extent, (vtx.0 - hw, vtx.1 - hw));
                    extend(&mut extent, (vtx.0 + hw, vtx.1 + hw));
                }
                LineJoin::Miter => {
                    let (d0, d1) = (dir((j + n - 1) % n), dir(j));
                    let sum = len((d0.0 + d1.0, d0.1 + d1.1));
                    let turn = (d1.0 - d0.0, d1.1 - d0.1);
                    let turn_len = len(turn);
                    if sum <= 1e-6 || turn_len <= 1e-6 {
                        continue;
                    }
                    let ratio = 2. / sum;
                    if ratio <= options.miter_limit {
                        let k = hw * ratio / turn_len;
                        extend(&mut extent, (vtx.0 - turn.0 * k, vtx.1 - turn.1 * k));
                    }
                }
            }
        }
    }
    extent_bounds(extent)
}
