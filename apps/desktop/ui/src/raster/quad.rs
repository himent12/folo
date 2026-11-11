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
}

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
        QuadPrep { sh, mode, fx0, fx1, split_x, fy0, fy1, split_y, bg, inside2, outside2 }
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
        // Column colours of free rows, one set per vertical half (computed on first use).
        let w = (r.x1 - r.x0) as usize;
        let mut cols: [Option<Box<[Src]>>; 2] = [None, None];
        for y in r.y0..r.y1 {
            let free_row = y >= self.fy0 && y < self.fy1;
            if free_row {
                let half = (y >= self.split_y) as usize;
                let rep = if half == 0 { self.fy0 } else { self.fy1 - 1 };
                let cs = cols[half].get_or_insert_with(|| {
                    let mut v = vec![Src::default(); w].into_boxed_slice();
                    for x in (r.x0..a).chain(b..r.x1) {
                        v[(x - r.x0) as usize] = self.eval(x, rep);
                    }
                    v
                });
                let row = t.row(y, r.x0, r.x1);
                let (left, rest) = row.split_at_mut((a - r.x0) as usize);
                let (mid, right) = rest.split_at_mut((b - a) as usize);
                for (p, s) in left.iter_mut().zip(cs.iter()) {
                    *p = s.over(*p);
                }
                fill_span(mid, &self.bg);
                for (p, s) in right.iter_mut().zip(cs[(b - r.x0) as usize..].iter()) {
                    *p = s.over(*p);
                }
            } else {
                let row = t.row(y, r.x0, r.x1);
                for x in (r.x0..a).chain(b..r.x1) {
                    let p = &mut row[(x - r.x0) as usize];
                    *p = self.eval(x, y).over(*p);
                }
                // One colour per horizontal half across the free columns, from the half's
                // outermost free column.
                let m = self.split_x.clamp(a, b);
                if a < m {
                    let s = self.eval(self.fx0, y);
                    fill_span(&mut row[(a - r.x0) as usize..(m - r.x0) as usize], &s);
                }
                if m < b {
                    let s = self.eval(self.fx1 - 1, y);
                    fill_span(&mut row[(m - r.x0) as usize..(b - r.x0) as usize], &s);
                }
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
