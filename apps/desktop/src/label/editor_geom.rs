//! Geometry and pointer-handle helpers for interactive bounding-box Move and Resize in Agent Labeling.

use folo_ui::{Bounds, CursorStyle, Pixels, Point, point, px};

use super::editor::{DRAG_THRESHOLD_PX, MIN_BOX_PX};
use crate::api::BBox;

/// The 8 interactive resize handles around a selected bounding box.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ResizeHandle {
    Nw,
    N,
    Ne,
    E,
    Se,
    S,
    Sw,
    W,
}

impl ResizeHandle {
    /// Cursor shown when hovering or dragging this handle.
    pub fn cursor(self) -> CursorStyle {
        match self {
            Self::Nw | Self::Se => CursorStyle::ResizeUpLeftDownRight,
            Self::Ne | Self::Sw => CursorStyle::ResizeUpRightDownLeft,
            Self::N | Self::S => CursorStyle::ResizeUpDown,
            Self::E | Self::W => CursorStyle::ResizeLeftRight,
        }
    }
}

/// Kind of interactive transform currently in progress on a selected box.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TransformKind {
    Move,
    Resize(ResizeHandle),
}

/// In-progress click-drag move or resize gesture on `draft[index]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoxTransform {
    pub index: usize,
    pub orig: BBox,
    pub start_img: (f32, f32),
    pub cur_img: (f32, f32),
    pub start_px: (f32, f32),
    pub cur_px: (f32, f32),
    pub kind: TransformKind,
    /// `LabelEditor::draft_gen` when the drag began; checked before preview/commit.
    pub start_draft_gen: u64,
    /// Becomes true once pointer travel exceeds [`DRAG_THRESHOLD_PX`].
    pub moved: bool,
}

impl BoxTransform {
    pub fn new(
        index: usize,
        orig: BBox,
        start_img: (f32, f32),
        start_px: (f32, f32),
        kind: TransformKind,
        start_draft_gen: u64,
    ) -> Self {
        Self {
            index,
            orig,
            start_img,
            cur_img: start_img,
            start_px,
            cur_px: start_px,
            kind,
            start_draft_gen,
            moved: false,
        }
    }

    /// Updates pointer coordinates; marks `moved` once travel passes `DRAG_THRESHOLD_PX`.
    pub fn update(&mut self, cur_img: (f32, f32), cur_px: (f32, f32)) {
        self.cur_img = cur_img;
        self.cur_px = cur_px;
        let dx = cur_px.0 - self.start_px.0;
        let dy = cur_px.1 - self.start_px.1;
        if dx * dx + dy * dy >= DRAG_THRESHOLD_PX * DRAG_THRESHOLD_PX {
            self.moved = true;
        }
    }

    /// Computes the clamped preview/final bounding box for this drag gesture using pointer delta
    /// from mouse-down so off-center grabs never jump.
    pub fn apply(&self, img_w: f32, img_h: f32) -> BBox {
        let dx = self.cur_img.0 - self.start_img.0;
        let dy = self.cur_img.1 - self.start_img.1;
        match self.kind {
            TransformKind::Move => apply_box_move(self.orig, dx, dy, img_w, img_h),
            TransformKind::Resize(handle) => apply_box_resize(self.orig, handle, (dx, dy), img_w, img_h),
        }
    }
}

/// Minimum resized box span in exact tenth-pixel units (`4.0 px = 40 tenths`).
const MIN_BOX_TENTHS: i64 = (MIN_BOX_PX as i64) * 10;

fn to_tenths(v: f32) -> i64 {
    (f64::from(v) * 10.0).round() as i64
}

fn from_tenths(t: i64) -> f32 {
    (t as f64 / 10.0) as f32
}

/// Translates `orig` by `(dx, dy)` while preserving its existing dimensions (including valid
/// sub-4px boxes) and clamping inside `[0, img_w] × [0, img_h]` in exact tenth-pixel coordinates.
pub fn apply_box_move(orig: BBox, dx: f32, dy: f32, img_w: f32, img_h: f32) -> BBox {
    if !dx.is_finite()
        || !dy.is_finite()
        || !img_w.is_finite()
        || !img_h.is_finite()
        || img_w <= 0.0
        || img_h <= 0.0
    {
        return orig;
    }
    let max_w = to_tenths(img_w).max(0);
    let max_h = to_tenths(img_h).max(0);
    let ox1 = to_tenths(orig.x1).clamp(0, max_w);
    let oy1 = to_tenths(orig.y1).clamp(0, max_h);
    let ox2 = to_tenths(orig.x2).clamp(ox1, max_w);
    let oy2 = to_tenths(orig.y2).clamp(oy1, max_h);
    let bw = (ox2 - ox1).clamp(0, max_w);
    let bh = (oy2 - oy1).clamp(0, max_h);
    let mut x1 = ox1.saturating_add(to_tenths(dx)).clamp(0, max_w - bw);
    let mut y1 = oy1.saturating_add(to_tenths(dy)).clamp(0, max_h - bh);
    let mut x2 = x1 + bw;
    let mut y2 = y1 + bh;
    // Preserve server-valid >= 1.0 px span after f32 conversion (e.g. 2.1_f32 - 1.1_f32 == 0.9999999).
    if (orig.x2 - orig.x1) >= 1.0 && (from_tenths(x2) - from_tenths(x1)) < 1.0 {
        if x2 < max_w {
            x2 += 1;
        } else if x1 > 0 {
            x1 -= 1;
        }
    }
    if (orig.y2 - orig.y1) >= 1.0 && (from_tenths(y2) - from_tenths(y1)) < 1.0 {
        if y2 < max_h {
            y2 += 1;
        } else if y1 > 0 {
            y1 -= 1;
        }
    }
    BBox {
        class: orig.class,
        x1: from_tenths(x1),
        y1: from_tenths(y1),
        x2: from_tenths(x2),
        y2: from_tenths(y2),
    }
}

/// Resizes `orig` by applying pointer `delta = (dx, dy)` to the edge(s) owned by `handle`, keeping
/// the opposite edge anchored, clamping to `[0, img_w] × [0, img_h]`, and enforcing [`MIN_BOX_PX`]
/// in exact tenth-pixel integer coordinates on the dragged axis (preventing IEEE-754 `f32`
/// snap-back on fractional edges or off-center handle grabs).
pub fn apply_box_resize(orig: BBox, handle: ResizeHandle, delta: (f32, f32), img_w: f32, img_h: f32) -> BBox {
    if !delta.0.is_finite()
        || !delta.1.is_finite()
        || !img_w.is_finite()
        || !img_h.is_finite()
        || img_w < MIN_BOX_PX
        || img_h < MIN_BOX_PX
    {
        return orig;
    }
    let max_w = to_tenths(img_w);
    let max_h = to_tenths(img_h);
    let mut x1 = to_tenths(orig.x1).clamp(0, max_w);
    let mut y1 = to_tenths(orig.y1).clamp(0, max_h);
    let mut x2 = to_tenths(orig.x2).clamp(0, max_w);
    let mut y2 = to_tenths(orig.y2).clamp(0, max_h);
    let dx = to_tenths(delta.0);
    let dy = to_tenths(delta.1);
    let min_x = MIN_BOX_TENTHS.min(max_w);
    let min_y = MIN_BOX_TENTHS.min(max_h);

    let moves_w = matches!(handle, ResizeHandle::Nw | ResizeHandle::W | ResizeHandle::Sw);
    let moves_e = matches!(handle, ResizeHandle::Ne | ResizeHandle::E | ResizeHandle::Se);
    let moves_n = matches!(handle, ResizeHandle::Nw | ResizeHandle::N | ResizeHandle::Ne);
    let moves_s = matches!(handle, ResizeHandle::Sw | ResizeHandle::S | ResizeHandle::Se);

    if moves_w {
        let hi = (x2 - min_x).max(0);
        x1 = x1.saturating_add(dx).clamp(0, hi);
    } else if moves_e {
        let lo = (x1 + min_x).min(max_w);
        x2 = x2.saturating_add(dx).clamp(lo, max_w);
    }
    if moves_n {
        let hi = (y2 - min_y).max(0);
        y1 = y1.saturating_add(dy).clamp(0, hi);
    } else if moves_s {
        let lo = (y1 + min_y).min(max_h);
        y2 = y2.saturating_add(dy).clamp(lo, max_h);
    }

    if x2 <= x1 || y2 <= y1 {
        return orig;
    }
    BBox {
        class: orig.class,
        x1: from_tenths(x1),
        y1: from_tenths(y1),
        x2: from_tenths(x2),
        y2: from_tenths(y2),
    }
}

/// Screen-space center positions of the 8 resize handles for `bbox` painted inside `rect`.
pub fn handle_points(
    bbox: BBox,
    rect: Bounds<Pixels>,
    dims: (f32, f32),
) -> [(ResizeHandle, Point<Pixels>); 8] {
    let (w, h) = (dims.0.max(1.0), dims.1.max(1.0));
    let sx = f32::from(rect.size.width) / w;
    let sy = f32::from(rect.size.height) / h;
    let x1 = f32::from(rect.left()) + bbox.x1 * sx;
    let y1 = f32::from(rect.top()) + bbox.y1 * sy;
    let x2 = f32::from(rect.left()) + bbox.x2 * sx;
    let y2 = f32::from(rect.top()) + bbox.y2 * sy;
    let mx = (x1 + x2) * 0.5;
    let my = (y1 + y2) * 0.5;
    [
        (ResizeHandle::Nw, point(px(x1), px(y1))),
        (ResizeHandle::Ne, point(px(x2), px(y1))),
        (ResizeHandle::Se, point(px(x2), px(y2))),
        (ResizeHandle::Sw, point(px(x1), px(y2))),
        (ResizeHandle::N, point(px(mx), px(y1))),
        (ResizeHandle::E, point(px(x2), px(my))),
        (ResizeHandle::S, point(px(mx), px(y2))),
        (ResizeHandle::W, point(px(x1), px(my))),
    ]
}

/// Returns the closest resize handle within `radius_px` logical pixels of `pos`, preferring corners
/// over edge midpoints when boxes are small.
pub fn hit_resize_handle(
    bbox: BBox,
    rect: Bounds<Pixels>,
    dims: (f32, f32),
    pos: Point<Pixels>,
    radius_px: f32,
) -> Option<ResizeHandle> {
    if rect.size.width <= px(0.) || rect.size.height <= px(0.) {
        return None;
    }
    let pts = handle_points(bbox, rect, dims);
    let r2 = radius_px * radius_px;
    let px_ = f32::from(pos.x);
    let py_ = f32::from(pos.y);
    let mut best: Option<(ResizeHandle, f32)> = None;
    for (i, (handle, pt)) in pts.into_iter().enumerate() {
        let dx = px_ - f32::from(pt.x);
        let dy = py_ - f32::from(pt.y);
        let d2 = dx * dx + dy * dy;
        if d2 <= r2 {
            // Slight bias for corners (indices 0..4) when a small box puts edges very close to corners.
            let score = if i < 4 { d2 * 0.85 } else { d2 };
            if best.is_none_or(|(_, bd)| score < bd) {
                best = Some((handle, score));
            }
        }
    }
    best.map(|(h, _)| h)
}

#[cfg(test)]
mod tests {
    use super::*;
    use folo_ui::size;

    #[test]
    fn apply_box_move_preserves_size_and_clamps_to_image_bounds() {
        let b = BBox { class: 1, x1: 50.0, y1: 60.0, x2: 150.0, y2: 140.0 };
        let moved = apply_box_move(b, 25.0, -10.0, 640.0, 480.0);
        assert_eq!(moved, BBox { class: 1, x1: 75.0, y1: 50.0, x2: 175.0, y2: 130.0 });

        // Moving past top-left clamps to (0, 0) while preserving 100×80 size.
        let clamped_tl = apply_box_move(b, -200.0, -300.0, 640.0, 480.0);
        assert_eq!(clamped_tl, BBox { class: 1, x1: 0.0, y1: 0.0, x2: 100.0, y2: 80.0 });

        // Moving past bottom-right clamps flush to image edge without squishing.
        let clamped_br = apply_box_move(b, 900.0, 900.0, 640.0, 480.0);
        assert_eq!(clamped_br, BBox { class: 1, x1: 540.0, y1: 400.0, x2: 640.0, y2: 480.0 });

        // Moving a valid sub-4px box preserves its exact dimensions instead of enlarging it.
        let tiny = BBox { class: 0, x1: 10.0, y1: 12.0, x2: 12.0, y2: 14.5 };
        assert_eq!(
            apply_box_move(tiny, 5.0, 3.0, 640.0, 480.0),
            BBox { class: 0, x1: 15.0, y1: 15.0, x2: 17.0, y2: 17.5 }
        );

        // Moving a 1.0px box by 1.1px must keep a server-valid >= 1.0 f32 span (never 0.9999999).
        let one_px = BBox { class: 0, x1: 0.0, y1: 0.0, x2: 1.0, y2: 1.0 };
        let moved_one = apply_box_move(one_px, 1.1, 1.1, 640.0, 480.0);
        assert!(moved_one.x2 - moved_one.x1 >= 1.0, "moved 1px box width must stay >= 1.0: {moved_one:?}");
        assert!(moved_one.y2 - moved_one.y1 >= 1.0, "moved 1px box height must stay >= 1.0: {moved_one:?}");

        // Extreme finite deltas saturate safely without overflowing i64 in debug builds.
        let extreme = apply_box_move(b, 1e30, -1e30, 640.0, 480.0);
        assert_eq!(extreme, BBox { class: 1, x1: 540.0, y1: 0.0, x2: 640.0, y2: 80.0 });

        // Non-finite delta is a safe no-op.
        assert_eq!(apply_box_move(b, f32::NAN, 10.0, 640.0, 480.0), b);
    }

    #[test]
    fn apply_box_resize_clamps_and_enforces_min_size_across_handles() {
        let b = BBox { class: 2, x1: 100.0, y1: 100.0, x2: 200.0, y2: 200.0 };
        assert_eq!(
            apply_box_resize(b, ResizeHandle::Se, (60.0, 40.0), 640.0, 480.0),
            BBox { class: 2, x1: 100.0, y1: 100.0, x2: 260.0, y2: 240.0 }
        );
        // Dragging NW past SE stops at MIN_BOX_PX instead of inverting.
        assert_eq!(
            apply_box_resize(b, ResizeHandle::Nw, (200.0, 200.0), 640.0, 480.0),
            BBox { class: 2, x1: 200.0 - MIN_BOX_PX, y1: 200.0 - MIN_BOX_PX, x2: 200.0, y2: 200.0 }
        );
        // Dragging E past image width clamps to 640.
        assert_eq!(
            apply_box_resize(b, ResizeHandle::E, (999.0, 50.0), 640.0, 480.0),
            BBox { class: 2, x1: 100.0, y1: 100.0, x2: 640.0, y2: 200.0 }
        );
        // Dragging N above 0 clamps to 0; horizontal delta on N has zero vertical effect.
        assert_eq!(
            apply_box_resize(b, ResizeHandle::N, (50.0, -150.0), 640.0, 480.0),
            BBox { class: 2, x1: 100.0, y1: 0.0, x2: 200.0, y2: 200.0 }
        );

        // Off-center grab on N handle + horizontal-only drag produces zero jump on y1.
        let mut tr = BoxTransform::new(
            0,
            b,
            (150.0, 104.0),
            (150.0, 104.0),
            TransformKind::Resize(ResizeHandle::N),
            1,
        );
        tr.update((180.0, 104.0), (180.0, 104.0));
        assert_eq!(tr.apply(640.0, 480.0), b, "horizontal drag on N must not move top edge");

        // Fractional f32 edges (0.2, 0.7, 4.4) clamp cleanly to MIN_BOX_PX without snapping back.
        for x1 in [0.2_f32, 0.7, 1.2, 4.4, 4.9] {
            let frac = BBox { class: 0, x1, y1: 10.0, x2: x1 + 50.0, y2: 60.0 };
            let clamped = apply_box_resize(frac, ResizeHandle::E, (-100.0, 0.0), 640.0, 480.0);
            assert!(
                (clamped.x2 - (x1 + MIN_BOX_PX)).abs() < 1e-3,
                "x1={x1} must hold at MIN_BOX_PX, not snap back: {clamped:?}"
            );
        }

        // Resizing horizontally on a server box whose height is < 4px still works without snapping back.
        let thin = BBox { class: 0, x1: 10.0, y1: 20.0, x2: 50.0, y2: 22.0 };
        let widened = apply_box_resize(thin, ResizeHandle::E, (20.0, 0.0), 640.0, 480.0);
        assert_eq!(widened, BBox { class: 0, x1: 10.0, y1: 20.0, x2: 70.0, y2: 22.0 });
    }

    #[test]
    fn hit_resize_handle_detects_corners_and_edges() {
        let b = BBox { class: 0, x1: 100.0, y1: 100.0, x2: 300.0, y2: 200.0 };
        let rect = Bounds::new(point(px(0.), px(0.)), size(px(640.), px(480.)));
        assert_eq!(
            hit_resize_handle(b, rect, (640., 480.), point(px(101.), px(99.)), 8.0),
            Some(ResizeHandle::Nw)
        );
        assert_eq!(
            hit_resize_handle(b, rect, (640., 480.), point(px(200.), px(201.)), 8.0),
            Some(ResizeHandle::S)
        );
        assert_eq!(
            hit_resize_handle(b, rect, (640., 480.), point(px(302.), px(150.)), 8.0),
            Some(ResizeHandle::E)
        );
        assert_eq!(hit_resize_handle(b, rect, (640., 480.), point(px(200.), px(150.)), 8.0), None);
    }
}
