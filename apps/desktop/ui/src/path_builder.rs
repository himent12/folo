// Adapted from GPUI (Apache-2.0): crates/gpui/src/path_builder.rs
//! `PathBuilder`: GPUI's path API (lyon's SVG-style builder: `move_to`, `line_to`, curves, `close`),
//! built exactly as GPUI builds it: the lyon path (transformed, dashed with lyon's path measurements),
//! tessellated by lyon (fill: even-odd; stroke: butt caps, miter joins, miter limit 4; tolerance 0.1)
//! into triangles in logical pixels. The renderer rasterizes those triangles as GPUI's path pipeline
//! does, so strokes keep lyon's joins (its miter limit, inner joins, folds) to the sample.

use crate::{
    Background, Bounds, FillRule, LineCap, LineJoin, PathData, Pixels, Point, ScaledPixels, Scene,
    StylePaint, point, px,
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

/// A built path (GPUI's `Path<Pixels>`; named `BuiltPath` because the scene primitive is `Path`): lyon's
/// triangles and the box of their vertices. `PathBuilder::build` returns `BuiltPath<Pixels>`.
#[derive(Clone, Debug, PartialEq)]
pub struct BuiltPath<P: Clone + Debug + Default + PartialEq = Pixels> {
    /// Three points per triangle.
    pub triangles: Vec<Point<P>>,
    /// The vertices' bounding box (GPUI's `Path::bounds`: the first vertex's point, grown by each
    /// triangle).
    pub bounds: Bounds<P>,
}

impl BuiltPath<Pixels> {
    /// This path in device pixels.
    pub fn scale(&self, factor: f32) -> BuiltPath<ScaledPixels> {
        BuiltPath {
            triangles: self.triangles.iter().map(|p| p.scale(factor)).collect(),
            bounds: self.bounds.scale(factor),
        }
    }
}

impl BuiltPath<ScaledPixels> {
    /// The scene's geometry for this path (a new id).
    pub fn to_path_data(&self) -> PathData {
        PathData::from_triangles(self.triangles.clone())
    }
}

/// Pushes a path as GPUI's `Window::paint_path` does: scaled to device pixels, the paint faded by the
/// element opacity, clipped by the content mask, skipped when its bounds are outside the mask.
pub fn paint_path(
    scene: &mut Scene,
    cx: &StylePaint,
    path: &BuiltPath<Pixels>,
    color: impl Into<Background>,
) {
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
}

/// Builds a filled or stroked [`BuiltPath`] (GPUI's `PathBuilder`).
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

    /// Builds the path as GPUI does: the lyon path (transformed), dashed, then tessellated.
    pub fn build(self) -> anyhow::Result<BuiltPath<Pixels>> {
        let mut b = lyon::path::Path::svg_builder();
        let lp = |p: Point<Pixels>| lyon::math::point(p.x.into(), p.y.into());
        for subpath in &self.subpaths {
            b.move_to(lp(subpath.start));
            for segment in &subpath.segments {
                match *segment {
                    Segment::Line(to) => {
                        b.line_to(lp(to));
                    }
                    Segment::Quadratic { ctrl, to } => {
                        b.quadratic_bezier_to(lp(ctrl), lp(to));
                    }
                    Segment::Cubic { ctrl1, ctrl2, to } => {
                        b.cubic_bezier_to(lp(ctrl1), lp(ctrl2), lp(to));
                    }
                }
            }
            if subpath.closed {
                b.close();
            }
        }
        let path = match self.transform {
            Some(Affine([a, b2, c, d, e, f])) => {
                b.build().transformed(&lyon::math::Transform::new(a, b2, c, d, e, f))
            }
            None => b.build(),
        };
        match self.style {
            PathStyle::Stroke(options) => tessellate_stroke(self.dash_array, &path, &options),
            PathStyle::Fill(options) => tessellate_fill(&path, &options),
        }
    }
}

/// GPUI's `tessellate_fill`.
fn tessellate_fill(path: &lyon::path::Path, options: &FillOptions) -> anyhow::Result<BuiltPath<Pixels>> {
    use lyon::tessellation::{BuffersBuilder, FillTessellator, FillVertex, VertexBuffers};
    let rule = match options.fill_rule {
        FillRule::NonZero => lyon::tessellation::FillRule::NonZero,
        FillRule::EvenOdd => lyon::tessellation::FillRule::EvenOdd,
    };
    let options =
        lyon::tessellation::FillOptions::default().with_fill_rule(rule).with_tolerance(options.tolerance);
    let mut buf: VertexBuffers<lyon::math::Point, u16> = VertexBuffers::new();
    FillTessellator::new()
        .tessellate_path(path, &options, &mut BuffersBuilder::new(&mut buf, |v: FillVertex| v.position()))
        .map_err(|e| anyhow::anyhow!("fill tessellation: {e:?}"))?;
    Ok(built_path(buf))
}

/// GPUI's `tessellate_stroke`: dashes cut along the whole path with lyon's measurements (each "on"
/// dash its own subpath), then stroked. A negative or non-finite dash length is an error. GPUI's loop
/// never ends when a whole period adds nothing to the position (all lengths zero, or too small for
/// `f32` at that length); here the cutting stops there, so an all-zero pattern draws nothing, which is
/// what GPUI's endless zero-length dashes would draw.
fn tessellate_stroke(
    dash_array: Option<Vec<Pixels>>,
    path: &lyon::path::Path,
    options: &StrokeOptions,
) -> anyhow::Result<BuiltPath<Pixels>> {
    use lyon::algorithms::measure::{PathMeasurements, SampleType};
    use lyon::tessellation::{BuffersBuilder, StrokeTessellator, StrokeVertex, VertexBuffers};
    let dashed;
    let path = if let Some(dash_array) = dash_array {
        if dash_array.iter().any(|d| !d.0.is_finite() || d.0 < 0.) {
            anyhow::bail!("dash lengths must be finite and not negative: {dash_array:?}");
        }
        let measurements = PathMeasurements::from_path(path, 0.01);
        let mut sampler = measurements.create_sampler(path, SampleType::Normalized);
        let mut builder = lyon::path::Path::builder();
        let total_length = sampler.length();
        if total_length.is_infinite() {
            anyhow::bail!("dashing a path of infinite length");
        }
        let mut pos = 0.;
        let mut dash_index = 0;
        let mut period_start = pos;
        while pos < total_length && !dash_array.is_empty() {
            let dash_length = dash_array[dash_index % dash_array.len()].0;
            let next_pos = (pos + dash_length).min(total_length);
            if dash_index % 2 == 0 {
                sampler.split_range(pos / total_length..next_pos / total_length, &mut builder);
            }
            pos = next_pos;
            dash_index += 1;
            if dash_index % dash_array.len() == 0 {
                if pos == period_start {
                    break;
                }
                period_start = pos;
            }
        }
        dashed = builder.build();
        &dashed
    } else {
        path
    };
    let cap = match options.line_cap {
        LineCap::Butt => lyon::tessellation::LineCap::Butt,
        LineCap::Square => lyon::tessellation::LineCap::Square,
        LineCap::Round => lyon::tessellation::LineCap::Round,
    };
    let join = match options.line_join {
        LineJoin::Miter => lyon::tessellation::LineJoin::Miter,
        LineJoin::Bevel => lyon::tessellation::LineJoin::Bevel,
        LineJoin::Round => lyon::tessellation::LineJoin::Round,
    };
    let options = lyon::tessellation::StrokeOptions::default()
        .with_line_width(options.line_width)
        .with_line_cap(cap)
        .with_line_join(join)
        .with_miter_limit(options.miter_limit.max(lyon::tessellation::StrokeOptions::MINIMUM_MITER_LIMIT))
        .with_tolerance(options.tolerance);
    let mut buf: VertexBuffers<lyon::math::Point, u16> = VertexBuffers::new();
    StrokeTessellator::new()
        .tessellate_path(path, &options, &mut BuffersBuilder::new(&mut buf, |v: StrokeVertex| v.position()))
        .map_err(|e| anyhow::anyhow!("stroke tessellation: {e:?}"))?;
    Ok(built_path(buf))
}

/// GPUI's `build_path`: the triangles of the index buffer, and their vertices' box grown from the
/// first vertex (an empty tessellation is an empty path at the origin).
fn built_path(buf: lyon::tessellation::VertexBuffers<lyon::math::Point, u16>) -> BuiltPath<Pixels> {
    let Some(first) = buf.vertices.first() else {
        return BuiltPath { triangles: Vec::new(), bounds: Bounds::default() };
    };
    let mut min = (first.x, first.y);
    let mut max = min;
    let mut triangles = Vec::with_capacity(buf.indices.len());
    for i in buf.indices.as_chunks::<3>().0 {
        for &v in i {
            let p = buf.vertices[v as usize];
            min = (min.0.min(p.x), min.1.min(p.y));
            max = (max.0.max(p.x), max.1.max(p.y));
            triangles.push(point(px(p.x), px(p.y)));
        }
    }
    let bounds = Bounds::from_corners(point(px(min.0), px(min.1)), point(px(max.0), px(max.1)));
    BuiltPath { triangles, bounds }
}
