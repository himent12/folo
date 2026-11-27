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
    ) -> Self {
        Self { index, orig, start_img, cur_img: start_img, start_px, cur_px: start_px, kind, moved: false }
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

    /// Computes the clamped preview/final bounding box for this drag gesture.
    pub fn apply(&self, img_w: f32, img_h: f32) -> BBox {
        match self.kind {
            TransformKind::Move => {
                let dx = self.cur_img.0 - self.start_img.0;
                let dy = self.cur_img.1 - self.start_img.1;
                apply_box_move(self.orig, dx, dy, img_w, img_h)
            }
            TransformKind::Resize(handle) => apply_box_resize(self.orig, handle, self.cur_img, img_w, img_h),
        }
    }
}

fn round_tenth(v: f32) -> f32 {
    (v * 10.0).round() / 10.0
}

/// Translates `orig` by `(dx, dy)` while preserving its width/height and clamping inside `[0, img_w] × [0, img_h]`.
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
    let bw = (orig.x2 - orig.x1).clamp(MIN_BOX_PX.min(img_w), img_w);
    let bh = (orig.y2 - orig.y1).clamp(MIN_BOX_PX.min(img_h), img_h);
    let max_x1 = (img_w - bw).max(0.0);
    let max_y1 = (img_h - bh).max(0.0);
    let x1 = round_tenth((orig.x1 + dx).clamp(0.0, max_x1));
    let y1 = round_tenth((orig.y1 + dy).clamp(0.0, max_y1));
    let x2 = round_tenth((x1 + bw).min(img_w));
    let y2 = round_tenth((y1 + bh).min(img_h));
    BBox { class: orig.class, x1, y1, x2, y2 }
}

/// Resizes `orig` by dragging `handle` toward `target` in image coordinates, keeping the opposite
/// edge anchored, clamping to `[0, img_w] × [0, img_h]`, and enforcing [`MIN_BOX_PX`].
pub fn apply_box_resize(
    orig: BBox,
    handle: ResizeHandle,
    target: (f32, f32),
    img_w: f32,
    img_h: f32,
) -> BBox {
    if !target.0.is_finite()
        || !target.1.is_finite()
        || !img_w.is_finite()
        || !img_h.is_finite()
        || img_w < MIN_BOX_PX
        || img_h < MIN_BOX_PX
    {
        return orig;
    }
    let mut x1 = orig.x1.clamp(0.0, img_w);
    let mut y1 = orig.y1.clamp(0.0, img_h);
    let mut x2 = orig.x2.clamp(0.0, img_w);
    let mut y2 = orig.y2.clamp(0.0, img_h);
    let tx = target.0.clamp(0.0, img_w);
    let ty = target.1.clamp(0.0, img_h);

    match handle {
        ResizeHandle::Nw => {
            x1 = tx.min(x2 - MIN_BOX_PX).max(0.0);
            y1 = ty.min(y2 - MIN_BOX_PX).max(0.0);
        }
        ResizeHandle::N => {
            y1 = ty.min(y2 - MIN_BOX_PX).max(0.0);
        }
        ResizeHandle::Ne => {
            x2 = tx.max(x1 + MIN_BOX_PX).min(img_w);
            y1 = ty.min(y2 - MIN_BOX_PX).max(0.0);
        }
        ResizeHandle::E => {
            x2 = tx.max(x1 + MIN_BOX_PX).min(img_w);
        }
        ResizeHandle::Se => {
            x2 = tx.max(x1 + MIN_BOX_PX).min(img_w);
            y2 = ty.max(y1 + MIN_BOX_PX).min(img_h);
        }
        ResizeHandle::S => {
            y2 = ty.max(y1 + MIN_BOX_PX).min(img_h);
        }
        ResizeHandle::Sw => {
            x1 = tx.min(x2 - MIN_BOX_PX).max(0.0);
            y2 = ty.max(y1 + MIN_BOX_PX).min(img_h);
        }
        ResizeHandle::W => {
            x1 = tx.min(x2 - MIN_BOX_PX).max(0.0);
        }
    }

    let (rx1, ry1, rx2, ry2) = (round_tenth(x1), round_tenth(y1), round_tenth(x2), round_tenth(y2));
    if (rx2 - rx1) < MIN_BOX_PX || (ry2 - ry1) < MIN_BOX_PX {
        return orig;
    }
    BBox { class: orig.class, x1: rx1, y1: ry1, x2: rx2, y2: ry2 }
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

        // Non-finite delta is a safe no-op.
        assert_eq!(apply_box_move(b, f32::NAN, 10.0, 640.0, 480.0), b);
    }

    #[test]
    fn apply_box_resize_clamps_and_enforces_min_size_across_handles() {
        let b = BBox { class: 2, x1: 100.0, y1: 100.0, x2: 200.0, y2: 200.0 };
        assert_eq!(
            apply_box_resize(b, ResizeHandle::Se, (260.0, 240.0), 640.0, 480.0),
            BBox { class: 2, x1: 100.0, y1: 100.0, x2: 260.0, y2: 240.0 }
        );
        // Dragging NW past SE stops at MIN_BOX_PX instead of inverting.
        assert_eq!(
            apply_box_resize(b, ResizeHandle::Nw, (300.0, 300.0), 640.0, 480.0),
            BBox { class: 2, x1: 200.0 - MIN_BOX_PX, y1: 200.0 - MIN_BOX_PX, x2: 200.0, y2: 200.0 }
        );
        // Dragging E past image width clamps to 640.
        assert_eq!(
            apply_box_resize(b, ResizeHandle::E, (999.0, 150.0), 640.0, 480.0),
            BBox { class: 2, x1: 100.0, y1: 100.0, x2: 640.0, y2: 200.0 }
        );
        // Dragging N above 0 clamps to 0.
        assert_eq!(
            apply_box_resize(b, ResizeHandle::N, (150.0, -50.0), 640.0, 480.0),
            BBox { class: 2, x1: 100.0, y1: 0.0, x2: 200.0, y2: 200.0 }
        );
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
