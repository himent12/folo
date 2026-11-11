//! Quads as spans. An axis-aligned quad's pixels split into a grid: columns whose colour does not
//! depend on x ("free" columns, away from the left and right edges) and rows whose colour does not
//! depend on y. A free row and a free column meet in the background; a free row is the background
//! between two narrow bands of per-column colours; a band row is one colour across its free columns.
//! Only the corner boxes, where neither is free, run the shader per pixel (`quad_eval.rs`).

use super::blend::{Src, fill_span};
use super::geom::{IRect, raster_span};
use super::gradient::GradientSpan;
use super::quad_eval::QuadShader;
use super::target::Target;
use crate::{BackgroundTag, BorderStyle, Quad};
use std::sync::OnceLock;

const AA: f32 = 0.5;

/// How a quad is drawn.
pub(crate) enum QuadMode {
    /// Solid background, solid border: the band decomposition with constant colours.
    Bands,
    /// A gradient background: the band decomposition, gradient spans inside, the shader at the edges.
    GradientBands(GradientSpan),
    /// Unrounded, unbordered gradient: spans everywhere.
    GradientFill(GradientSpan),
    /// Dashed borders or the slash pattern: the shader at every pixel.
    PerPixel,
}

/// A quad ready for tiles: the shader, the free column and row ranges, and the background.
pub(crate) struct QuadPrep {
    pub sh: QuadShader,
    pub mode: QuadMode,
    /// Free columns `fx0..fx1`; `split_x` is the first column of the right half.
    fx0: i32,
    fx1: i32,
    split_x: i32,
    fy0: i32,
    fy1: i32,
    split_y: i32,
    /// The solid background.
    bg: Src,
    /// Per corner (tl, tr, br, bl): squared distances from the corner circle's centre within which a
    /// pixel is certainly the background, and beyond which it is certainly untouched.
    inside2: [f32; 4],
    outside2: [f32; 4],
    /// The rasterized span (`x0, y0, x1, y1`), before clipping.
    span: [i32; 4],
    /// The straight bands' colours, computed by the first tile that needs them.
    edges: OnceLock<Option<Edges>>,
}

/// The colours of a quad's straight bands: the columns left and right of the free columns (on free
/// rows, per vertical half), and the rows above and below the free rows (across the free columns,
/// per horizontal half). Computed once per frame and shared by every tile of the quad.
struct Edges {
    /// Per vertical half: columns `x0..fx0`, then `fx1..x1`.
    cols: [Vec<Src>; 2],
    /// Rows `y0..fy0`, then `fy1..y1`: the left and right halves' colours.
    rows: Vec<[Src; 2]>,
}

/// Bands wider than this (huge radii) are not tabulated; their tiles evaluate each pixel.
const MAX_BAND: i32 = 1024;

/// Slack for the corner shortcuts, far above the rounding of the shader's own maths.
const EPS: f32 = 1e-2;

/// Distance of a pixel centre's coordinate past the quad's edge on its side (`ctp`), and whether it
/// is in the low half (left / top).
#[inline]
fn ctp(i: i32, origin: f32, size: f32) -> (f32, bool) {
    let half = size / 2.;
    let c2p = (i as f32 + 0.5 - origin) - half;
    (c2p.abs() - half, c2p < 0.)
}

/// The free range along one axis: indices whose `ctp` is below `-t_low` (low half) or `-t_high`.
/// Returns `(f0, f1, split)` with the free set `f0..f1`.
fn free_range(origin: f32, size: f32, t_low: f32, t_high: f32) -> (i32, i32, i32) {
    let (lo, hi) = raster_span(origin, origin + size);
    let free = |i: i32| {
        let (c, low) = ctp(i, origin, size);
        c < -(if low { t_low } else { t_high })
    };
    let low_half = |i: i32| ctp(i, origin, size).1;
    // The first index of the high half.
    let mut split = ((origin + size / 2. - 0.5).ceil() as i32).clamp(lo, hi);
    while split > lo && !low_half(split - 1) {
        split -= 1;
    }
    while split < hi && low_half(split) {
        split += 1;
    }
    // Low half: free is monotone, rising toward the centre.
    let mut f0 = ((origin + t_low - 0.5).floor() as i32 + 1).clamp(lo, split);
    while f0 > lo && free(f0 - 1) {
        f0 -= 1;
    }
    while f0 < split && !free(f0) {
        f0 += 1;
    }
    // High half: free falls toward the far edge.
    let mut f1 = ((origin + size - t_high - 0.5).ceil() as i32).clamp(split, hi);
    while f1 < hi && free(f1) {
        f1 += 1;
    }
    while f1 > split && !free(f1 - 1) {
        f1 -= 1;
    }
    if f0 == split && f1 == split {
        return (split, split, split);
    }
    (f0, f1, split)
}

impl QuadPrep {
    pub fn new(sh: QuadShader, grad: Option<GradientSpan>) -> Self {
        let [r_tl, r_tr, r_br, r_bl] = sh.radii;
        let [w_t, w_r, w_b, w_l] = sh.widths;
        let reduced = |w: f32| if w == 0. { -AA } else { w };
        let t = |r: f32, w: f32| r.max(reduced(w) + AA).max(AA);
        let (fx0, fx1, split_x) =
            free_range(sh.origin[0], sh.size[0], t(r_tl.max(r_bl), w_l), t(r_tr.max(r_br), w_r));
        let (fy0, fy1, split_y) =
            free_range(sh.origin[1], sh.size[1], t(r_tl.max(r_tr), w_t), t(r_bl.max(r_br), w_b));
        let simple = sh.unrounded() && sh.unbordered();
        let mode = if sh.dashed || !(sh.paint.is_solid() || sh.paint.is_gradient()) {
            QuadMode::PerPixel
        } else if let Some(g) = grad {
            if simple { QuadMode::GradientFill(g) } else { QuadMode::GradientBands(g) }
        } else {
            QuadMode::Bands
        };
        let bg = Src::new(sh.paint.solid);
        // A corner's borders (x side, y side): left/right then top/bottom.
        let sides = [(w_l, w_t), (w_r, w_t), (w_r, w_b), (w_l, w_b)];
        let mut inside2 = [-1.; 4];
        let mut outside2 = [f32::INFINITY; 4];
        for (k, &(bx, by)) in sides.iter().enumerate() {
            let r = sh.radii[k];
            if r <= 0. {
                continue;
            }
            outside2[k] = (r + AA + EPS) * (r + AA + EPS);
            let (rx, ry) = (reduced(bx), reduced(by));
            let inner = r - rx.max(0.) - AA - EPS;
            if rx == ry && inner > 0. {
                inside2[k] = inner * inner;
            }
        }
        let (x0, x1) = raster_span(sh.origin[0], sh.origin[0] + sh.size[0]);
        let (y0, y1) = raster_span(sh.origin[1], sh.origin[1] + sh.size[1]);
        QuadPrep {
            sh,
            mode,
            fx0,
            fx1,
            split_x,
            fy0,
            fy1,
            split_y,
            bg,
            inside2,
            outside2,
            span: [x0, y0, x1, y1],
            edges: OnceLock::new(),
        }
    }

    /// The representative free row of vertical half `half` (column colours are the same on every
    /// free row of a half).
    fn rep_row(&self, half: usize) -> i32 {
        if half == 0 { self.fy0 } else { self.fy1 - 1 }
    }

    fn edges(&self) -> Option<&Edges> {
        self.edges
            .get_or_init(|| {
                let [x0, y0, x1, y1] = self.span;
                let bands = [self.fx0 - x0, x1 - self.fx1, self.fy0 - y0, y1 - self.fy1];
                if bands.iter().any(|&b| b > MAX_BAND) {
                    return None;
                }
                let has_rows = [self.fy0 < self.fy1.min(self.split_y), self.fy0.max(self.split_y) < self.fy1];
                let cols = [0, 1].map(|half| {
                    if !has_rows[half] {
                        return Vec::new();
                    }
                    let rep = self.rep_row(half);
                    (x0..self.fx0).chain(self.fx1..x1).map(|x| self.eval(x, rep)).collect()
                });
                let (left, right) = (self.fx0 < self.split_x, self.split_x < self.fx1);
                let rows = (y0..self.fy0)
                    .chain(self.fy1..y1)
                    .map(|y| {
                        [
                            if left { self.eval(self.fx0, y) } else { Src::NOOP },
                            if right { self.eval(self.fx1 - 1, y) } else { Src::NOOP },
                        ]
                    })
                    .collect();
                Some(Edges { cols, rows })
            })
            .as_ref()
    }

    /// The shader's output at a pixel as a blendable source. Inside a rounded corner, pixels well
    /// inside the inner edge are the background and pixels past the outer edge are untouched (the
    /// shader's own branches, decided from the distance to the corner's centre); only the ring
    /// between runs the shader.
    fn eval(&self, x: i32, y: i32) -> Src {
        if matches!(self.mode, QuadMode::Bands) {
            let sh = &self.sh;
            let half = [sh.size[0] / 2., sh.size[1] / 2.];
            let c2p = [x as f32 + 0.5 - sh.origin[0] - half[0], y as f32 + 0.5 - sh.origin[1] - half[1]];
            let k = match (c2p[0] < 0., c2p[1] < 0.) {
                (true, true) => 0,
                (false, true) => 1,
                (false, false) => 2,
                (true, false) => 3,
            };
            let r = sh.radii[k];
            let cx = c2p[0].abs() - half[0] + r;
            let cy = c2p[1].abs() - half[1] + r;
            if r > 0. && cx >= 0. && cy >= 0. {
                let d2 = cx * cx + cy * cy;
                if d2 >= self.outside2[k] {
                    return Src::NOOP;
                }
                if d2 <= self.inside2[k] {
                    return self.bg;
                }
            }
        }
        Src::new(self.sh.eval(x, y))
    }

    /// Draws the quad's pixels inside `r` (already clipped to the tile and the quad's clip).
    pub fn draw(&self, t: &mut Target, r: IRect) {
        match &self.mode {
            QuadMode::PerPixel => {
                for y in r.y0..r.y1 {
                    let row = t.row(y, r.x0, r.x1);
                    for (x, p) in (r.x0..r.x1).zip(row.iter_mut()) {
                        *p = self.eval(x, y).over(*p);
                    }
                }
            }
            QuadMode::GradientFill(g) => {
                for y in r.y0..r.y1 {
                    g.draw(t.row(y, r.x0, r.x1), r.x0, y);
                }
            }
            QuadMode::GradientBands(g) => self.draw_gradient_bands(t, r, g),
            QuadMode::Bands => self.draw_bands(t, r),
        }
    }

    /// The free columns of `r` (clamped).
    fn free_cols(&self, r: &IRect) -> (i32, i32) {
        let a = self.fx0.clamp(r.x0, r.x1);
        (a, self.fx1.clamp(a, r.x1))
    }

    fn draw_bands(&self, t: &mut Target, r: IRect) {
        let (a, b) = self.free_cols(&r);
        let edges = self.edges();
        let [x0, y0, ..] = self.span;
        // Index of a band column / row in `Edges`.
        let col = |x: i32| (if x < self.fx0 { x - x0 } else { self.fx0 - x0 + x - self.fx1 }) as usize;
        let row_of = |y: i32| (if y < self.fy0 { y - y0 } else { self.fy0 - y0 + y - self.fy1 }) as usize;
        for y in r.y0..r.y1 {
            let row = t.row(y, r.x0, r.x1);
            if y >= self.fy0 && y < self.fy1 {
                // A free row: band columns from the table, the background between.
                let half = (y >= self.split_y) as usize;
                for x in (r.x0..a).chain(b..r.x1) {
                    let s = match edges {
                        Some(e) => e.cols[half][col(x)],
                        None => self.eval(x, self.rep_row(half)),
                    };
                    let p = &mut row[(x - r.x0) as usize];
                    *p = s.over(*p);
                }
                fill_span(&mut row[(a - r.x0) as usize..(b - r.x0) as usize], &self.bg);
            } else {
                // A band row: the corner boxes per pixel, one colour per half across the free
                // columns (from the half's outermost free column).
                for x in (r.x0..a).chain(b..r.x1) {
                    let p = &mut row[(x - r.x0) as usize];
                    *p = self.eval(x, y).over(*p);
                }
                let m = self.split_x.clamp(a, b);
                let colours = match edges {
                    Some(e) => e.rows[row_of(y)],
                    None => [
                        if a < m { self.eval(self.fx0, y) } else { Src::NOOP },
                        if m < b { self.eval(self.fx1 - 1, y) } else { Src::NOOP },
                    ],
                };
                fill_span(&mut row[(a - r.x0) as usize..(m - r.x0) as usize], &colours[0]);
                fill_span(&mut row[(m - r.x0) as usize..(b - r.x0) as usize], &colours[1]);
            }
        }
    }

    fn draw_gradient_bands(&self, t: &mut Target, r: IRect, g: &GradientSpan) {
        let (a, b) = self.free_cols(&r);
        for y in r.y0..r.y1 {
            let free_row = y >= self.fy0 && y < self.fy1;
            let row = t.row(y, r.x0, r.x1);
            if free_row {
                for x in (r.x0..a).chain(b..r.x1) {
                    let p = &mut row[(x - r.x0) as usize];
                    *p = self.eval(x, y).over(*p);
                }
                g.draw(&mut row[(a - r.x0) as usize..(b - r.x0) as usize], a, y);
            } else {
                for (x, p) in (r.x0..r.x1).zip(row.iter_mut()) {
                    *p = self.eval(x, y).over(*p);
                }
            }
        }
    }
}

/// Pixels where a quad certainly writes its opaque background and nothing else (a conservative
/// inset of the free rows by free columns), for occlusion. `rect` is the quad's pixel rectangle.
pub(crate) fn opaque_core(q: &Quad, rect: IRect) -> IRect {
    let bg = &q.background;
    if bg.tag != BackgroundTag::Solid || bg.solid.a < 1. {
        return IRect::EMPTY;
    }
    let (r, w) = (&q.corner_radii, &q.border_widths);
    let simple = [r.top_left, r.top_right, r.bottom_right, r.bottom_left, w.top, w.right, w.bottom, w.left]
        .iter()
        .all(|v| v.0 == 0.);
    if simple {
        // The fast path writes the background at every rasterized pixel.
        return rect;
    }
    if q.border_style == BorderStyle::Dashed {
        return IRect::EMPTY;
    }
    let reduced = |w: f32| if w == 0. { -AA } else { w };
    let t = |r: f32, w: f32| r.max(reduced(w) + AA).max(AA) + 1.;
    let (o, s) = (q.bounds.origin, q.bounds.size);
    let x0 = (o.x.0 + t(r.top_left.0.max(r.bottom_left.0), w.left.0)).ceil();
    let x1 = (o.x.0 + s.width.0 - t(r.top_right.0.max(r.bottom_right.0), w.right.0)).floor();
    let y0 = (o.y.0 + t(r.top_left.0.max(r.top_right.0), w.top.0)).ceil();
    let y1 = (o.y.0 + s.height.0 - t(r.bottom_left.0.max(r.bottom_right.0), w.bottom.0)).floor();
    IRect::new(x0 as i32, y0 as i32, x1 as i32, y1 as i32).intersect(&rect)
}
