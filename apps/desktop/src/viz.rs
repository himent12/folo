//! Images with detection boxes, painted on a canvas.
//!
//! Boxes follow design/README.md: a 2 px stroke in the class colour wrapped by a 1 px halo so it
//! reads on any photo, square corners (boxes are measurements), and a label chip on the box's
//! top-left corner outside the box, moved inside when it would leave the image.

use std::sync::Arc;

use gpui::{
    App, BorderStyle, Bounds, Corners, Edges, FontWeight, Hsla, Pixels, Point, RenderImage, SharedString,
    TextRun, Window, fill, point, px, quad, size, transparent_black,
};

use crate::theme::{SANS, Theme, alpha, tokens, ty};

#[derive(Clone, Debug)]
pub struct DrawBox {
    pub class: u32,
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub label: Option<SharedString>,
    pub score: Option<f32>,
    /// Ground truth drawn as a dashed-look thin outline instead of a solid detection box.
    pub truth: bool,
    pub highlight: bool,
    pub dim: bool,
}

impl DrawBox {
    pub fn new(class: u32, x1: f32, y1: f32, x2: f32, y2: f32) -> Self {
        Self { class, x1, y1, x2, y2, label: None, score: None, truth: false, highlight: false, dim: false }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Fit {
    /// Fill the bounds, cropping overflow (grid tiles).
    Cover,
    /// Fit inside the bounds, letterboxed (inspector, playground).
    Contain,
}

/// Where an image of `w` × `h` lands inside `b` for a fit, zoom (1 = fit) and pan (px).
pub fn image_rect(b: Bounds<Pixels>, w: f32, h: f32, fit: Fit, zoom: f32, pan: (f32, f32)) -> Bounds<Pixels> {
    let (bw, bh) = (f32::from(b.size.width), f32::from(b.size.height));
    let s = match fit {
        Fit::Cover => (bw / w).max(bh / h),
        Fit::Contain => (bw / w).min(bh / h),
    } * zoom;
    let (dw, dh) = (w * s, h * s);
    // Pan is held so at least a 48 px strip of the image stays in view, however far it's pushed.
    let keep = |view: f32, drawn: f32| ((view + drawn) / 2. - 48.).max(0.);
    let pan = (pan.0.clamp(-keep(bw, dw), keep(bw, dw)), pan.1.clamp(-keep(bh, dh), keep(bh, dh)));
    let x = f32::from(b.left()) + (bw - dw) / 2. + pan.0;
    let y = f32::from(b.top()) + (bh - dh) / 2. + pan.1;
    Bounds::new(point(px(x), px(y)), size(px(dw), px(dh)))
}

fn run(text: &str, family: &'static str, weight: FontWeight, color: Hsla) -> TextRun {
    TextRun {
        len: text.len(),
        font: gpui::Font {
            weight,
            features: gpui::FontFeatures(Arc::new(vec![("tnum".into(), 1)])),
            ..gpui::font(family)
        },
        color,
        background_color: None,
        underline: None,
        strikethrough: None,
    }
}

/// Paint an image (or a placeholder well while it loads) with boxes in image pixel coordinates.
#[allow(clippy::too_many_arguments)]
pub fn paint_image_with_boxes(
    t: &Theme,
    window: &mut Window,
    cx: &mut App,
    bounds: Bounds<Pixels>,
    img: Option<&Arc<RenderImage>>,
    (w, h): (f32, f32),
    boxes: &[DrawBox],
    fit: Fit,
    zoom: f32,
    pan: (f32, f32),
    labels: bool,
    corner: f32,
) {
    let rect = image_rect(bounds, w.max(1.), h.max(1.), fit, zoom, pan);
    window.with_content_mask(Some(gpui::ContentMask { bounds }), |window| {
        match img {
            Some(img) => {
                let r = if fit == Fit::Cover { corner } else { 0. };
                let _ = window.paint_image(rect, Corners::all(px(r)), img.clone(), 0, false);
            }
            None => window.paint_quad(fill(rect, t.bg_raised)),
        }
        let sx = f32::from(rect.size.width) / w.max(1.);
        let sy = f32::from(rect.size.height) / h.max(1.);
        let to = |x: f32, y: f32| point(rect.left() + px(x * sx), rect.top() + px(y * sy));
        let halo = rgba(tokens::CLASS_HALO);
        let chip_text = rgba(tokens::CLASS_CHIP_TEXT);
        // Dimmed boxes first, highlighted last, so focus sits on top.
        let mut order: Vec<&DrawBox> = boxes.iter().collect();
        order.sort_by_key(|b| (b.highlight as u8, !b.dim as u8));
        // Label chips already placed, so later chips can step around them.
        let mut placed: Vec<Bounds<Pixels>> = Vec::new();
        // Two passes: every box first, then every label chip, so no box stroke covers a chip.
        for pass in 0..2 {
            for &b in &order {
                let color = t.class(b.class);
                let a = if b.dim { 0.3 } else { 1.0 };
                let p1 = to(b.x1, b.y1);
                let p2 = to(b.x2, b.y2);
                let r = Bounds::from_corners(p1, p2);
                if pass == 0 {
                    if b.truth {
                        // Ground truth: thin outline in the class colour over a halo, no label chip.
                        window.paint_quad(quad(
                            r.dilate(px(1.)),
                            px(0.),
                            transparent_black(),
                            px(1.),
                            alpha(halo, 0.6 * a),
                            BorderStyle::Solid,
                        ));
                        window.paint_quad(quad(
                            r,
                            px(0.),
                            transparent_black(),
                            px(1.),
                            alpha(color, 0.9 * a),
                            BorderStyle::Dashed,
                        ));
                        continue;
                    }
                    if b.highlight {
                        window.paint_quad(fill(r, alpha(color, tokens::CLASS_FILL_OPACITY * 1.6)));
                    }
                    let (stroke, ink) = if b.highlight { (3., t.accent_solid) } else { (2., color) };
                    window.paint_quad(quad(
                        r.dilate(px(1.)),
                        px(0.),
                        transparent_black(),
                        px(1.),
                        alpha(halo, a * 0.8),
                        BorderStyle::Solid,
                    ));
                    window.paint_quad(quad(
                        r,
                        px(0.),
                        transparent_black(),
                        Edges::all(px(stroke)),
                        alpha(ink, a),
                        BorderStyle::Solid,
                    ));
                }
                // Boxes under 20 px on screen show their label only when selected (design/README).
                let small = (r.size.height < px(20.) || r.size.width < px(20.)) && !b.highlight;
                // Cropped out of view (Fill): no chip floating over someone else's pixels.
                let visible = r.intersect(&bounds);
                let off_view = visible.size.width < px(4.) || visible.size.height < px(4.);
                if pass == 1
                    && !b.truth
                    && labels
                    && !small
                    && !off_view
                    && (b.label.is_some() || b.score.is_some())
                {
                    // Small views (< 480 px): 14 px chips with 10 px text (design/README).
                    let compact = rect.size.width < px(480.);
                    let fs = if compact { 10. } else { ty::XS.size };
                    let name = b.label.clone().unwrap_or_default();
                    // The selected box's chip takes the selection colour, like its outline.
                    let (chip_bg, ink) = if b.highlight {
                        (t.accent_solid, t.accent_on)
                    } else if b.dim {
                        // Faded with their boxes, so the selection is the brightest thing.
                        (alpha(color, 0.3), alpha(chip_text, 0.6))
                    } else {
                        (color, chip_text)
                    };
                    let score = b.score.map(|s| format!("{s:.2}")).unwrap_or_default();
                    let name_line = window.text_system().shape_line(
                        name.clone(),
                        px(fs),
                        &[run(&name, SANS, FontWeight::MEDIUM, ink)],
                        None,
                    );
                    let score_line = window.text_system().shape_line(
                        score.clone().into(),
                        px(fs),
                        &[run(&score, SANS, FontWeight::NORMAL, ink)],
                        None,
                    );
                    let gap = if score.is_empty() || name.is_empty() { px(0.) } else { px(5.) };
                    let cw = name_line.width + gap + score_line.width + px(10.);
                    let ch = px(if compact { 14. } else { 16. });
                    // Preferred spots: above the top-left corner, inside it, below the bottom-left
                    // corner, inside the bottom-left. Take the first that is on the image and free.
                    // Within the visible part of its box (and the view).
                    let clamp_x = |x: Pixels| {
                        x.min(visible.right() - cw)
                            .min(bounds.right() - cw)
                            .max(visible.left())
                            .max(bounds.left())
                    };
                    let candidates = [
                        point(clamp_x(r.left() - px(1.)), r.top() - ch - px(1.)),
                        point(clamp_x(r.left() + px(1.)), r.top() + px(1.)),
                        point(clamp_x(r.left() - px(1.)), r.bottom() + px(1.)),
                        point(clamp_x(r.left() + px(1.)), r.bottom() - ch - px(1.)),
                    ];
                    let fits = |o: Point<Pixels>| {
                        o.y >= bounds.top().max(rect.top()) && o.y + ch <= bounds.bottom().min(rect.bottom())
                    };
                    let free = |c: &Bounds<Pixels>| !placed.iter().any(|p| p.dilate(px(2.)).intersects(c));
                    let origin = candidates
                        .iter()
                        .copied()
                        .find(|o| fits(*o) && free(&Bounds::new(*o, size(cw, ch))))
                        .or_else(|| candidates.iter().copied().find(|o| fits(*o)))
                        .unwrap_or(candidates[1]);
                    let chip = Bounds::new(origin, size(cw, ch));
                    placed.push(chip);
                    // Dimming applies to strokes only: chip text keeps its contrast.
                    window.paint_quad(fill(chip, chip_bg).corner_radii(px(tokens::radius::XS)));
                    let (ty_, lh) =
                        if compact { (origin.y + px(1.), px(12.)) } else { (origin.y + px(1.5), px(13.)) };
                    let _ = name_line.paint(point(origin.x + px(5.), ty_), lh, window, cx);
                    let _ = score_line.paint(
                        point(origin.x + px(5.) + name_line.width + gap, ty_),
                        lh,
                        window,
                        cx,
                    );
                }
            }
        }
    });
}

fn rgba(v: u32) -> Hsla {
    gpui::rgba(v).into()
}

/// Zoom to `new_zoom` keeping the image point under `cursor` fixed; returns the new pan.
/// Zoom by `f` about the view centre, within 1×..=12×. Pan scales by the zoom actually applied,
/// so pressing zoom-in at the cap leaves the view where it is.
pub fn zoom_step(zoom: f32, pan: (f32, f32), f: f32) -> (f32, (f32, f32)) {
    let z = (zoom * f).clamp(1.0, 12.0);
    if z <= 1.0 {
        return (1.0, (0., 0.));
    }
    let k = z / zoom;
    (z, (pan.0 * k, pan.1 * k))
}

pub fn zoom_about(
    bounds: Bounds<Pixels>,
    (w, h): (f32, f32),
    zoom: f32,
    pan: (f32, f32),
    cursor: Point<Pixels>,
    new_zoom: f32,
) -> (f32, f32) {
    if new_zoom <= 1.0 {
        return (0., 0.);
    }
    let rect = image_rect(bounds, w, h, Fit::Contain, zoom, pan);
    let fx = f32::from(cursor.x - rect.left()) / f32::from(rect.size.width);
    let fy = f32::from(cursor.y - rect.top()) / f32::from(rect.size.height);
    let fresh = image_rect(bounds, w, h, Fit::Contain, new_zoom, (0., 0.));
    let want_x = f32::from(cursor.x) - fx * f32::from(fresh.size.width);
    let want_y = f32::from(cursor.y) - fy * f32::from(fresh.size.height);
    (want_x - f32::from(fresh.left()), want_y - f32::from(fresh.top()))
}

/// Map a window point to image pixel coordinates (for hover hit-testing).
pub fn to_image(
    bounds: Bounds<Pixels>,
    (w, h): (f32, f32),
    fit: Fit,
    zoom: f32,
    pan: (f32, f32),
    p: Point<Pixels>,
) -> (f32, f32) {
    let rect = image_rect(bounds, w, h, fit, zoom, pan);
    let x = f32::from(p.x - rect.left()) / f32::from(rect.size.width) * w;
    let y = f32::from(p.y - rect.top()) / f32::from(rect.size.height) * h;
    (x, y)
}

/// Index of the smallest box containing an image point.
pub fn hit_box(boxes: &[DrawBox], x: f32, y: f32) -> Option<usize> {
    boxes
        .iter()
        .enumerate()
        .filter(|(_, b)| x >= b.x1 && x <= b.x2 && y >= b.y1 && y <= b.y2)
        .min_by(|a, b| {
            ((a.1.x2 - a.1.x1) * (a.1.y2 - a.1.y1)).total_cmp(&((b.1.x2 - b.1.x1) * (b.1.y2 - b.1.y1)))
        })
        .map(|(i, _)| i)
}

#[cfg(test)]
mod tests {
    #[test]
    fn zoom_step_holds_still_at_the_cap() {
        use super::zoom_step;
        assert_eq!(zoom_step(2.0, (10., -4.), 2.0), (4.0, (20., -8.)));
        // At 12× a further zoom-in changes nothing.
        assert_eq!(zoom_step(12.0, (30., 6.), 1.4), (12.0, (30., 6.)));
        // Approaching the cap scales pan by the zoom actually applied.
        let (z, p) = zoom_step(10.0, (10., 0.), 1.4);
        assert_eq!(z, 12.0);
        assert!((p.0 - 12.0).abs() < 1e-4);
        // Back to fit recentres.
        assert_eq!(zoom_step(1.2, (5., 5.), 0.5), (1.0, (0., 0.)));
    }

    use super::*;

    #[test]
    fn contain_letterboxes_and_cover_crops() {
        let b = Bounds::new(point(px(0.), px(0.)), size(px(200.), px(100.)));
        let c = image_rect(b, 100., 100., Fit::Contain, 1.0, (0., 0.));
        assert_eq!((f32::from(c.size.width), f32::from(c.left())), (100., 50.));
        let v = image_rect(b, 100., 100., Fit::Cover, 1.0, (0., 0.));
        assert_eq!((f32::from(v.size.width), f32::from(v.top())), (200., -50.));
    }

    #[test]
    fn mapping_round_trips() {
        let b = Bounds::new(point(px(10.), px(20.)), size(px(400.), px(300.)));
        let (x, y) = to_image(b, (640., 480.), Fit::Contain, 2.0, (13., -7.), {
            let r = image_rect(b, 640., 480., Fit::Contain, 2.0, (13., -7.));
            point(r.left() + r.size.width / 4., r.top() + r.size.height / 2.)
        });
        assert!((x - 160.).abs() < 0.01 && (y - 240.).abs() < 0.01);
    }

    #[test]
    fn zoom_keeps_the_cursor_point() {
        let b = Bounds::new(point(px(0.), px(0.)), size(px(400.), px(300.)));
        let cursor = point(px(130.), px(90.));
        let before = to_image(b, (640., 480.), Fit::Contain, 1.0, (0., 0.), cursor);
        let pan = zoom_about(b, (640., 480.), 1.0, (0., 0.), cursor, 3.0);
        let after = to_image(b, (640., 480.), Fit::Contain, 3.0, pan, cursor);
        assert!((before.0 - after.0).abs() < 0.01 && (before.1 - after.1).abs() < 0.01);
    }

    #[test]
    fn hit_prefers_smallest() {
        let boxes = vec![DrawBox::new(0, 0., 0., 100., 100.), DrawBox::new(1, 10., 10., 20., 20.)];
        assert_eq!(hit_box(&boxes, 15., 15.), Some(1));
        assert_eq!(hit_box(&boxes, 50., 50.), Some(0));
        assert_eq!(hit_box(&boxes, 150., 50.), None);
    }
}
