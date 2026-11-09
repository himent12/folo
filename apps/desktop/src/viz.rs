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
    /// One of a set that shares one chip: the first of them the painter can label (largest first,
    /// as placement runs) takes it, and the rest stay unnamed (a filtered thumbnail names one match).
    pub one_of: bool,
}

impl DrawBox {
    pub fn new(class: u32, x1: f32, y1: f32, x2: f32, y2: f32) -> Self {
        Self {
            class,
            x1,
            y1,
            x2,
            y2,
            label: None,
            score: None,
            truth: false,
            highlight: false,
            dim: false,
            one_of: false,
        }
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
/// `pan` held so at least a 48 px strip of the image stays in view. Viewers store the clamped
/// value back, so reversing a pan never has a dead zone.
pub fn clamp_pan(b: Bounds<Pixels>, w: f32, h: f32, fit: Fit, zoom: f32, pan: (f32, f32)) -> (f32, f32) {
    let (bw, bh) = (f32::from(b.size.width), f32::from(b.size.height));
    let s = match fit {
        Fit::Cover => (bw / w).max(bh / h),
        Fit::Contain => (bw / w).min(bh / h),
    } * zoom;
    let (dw, dh) = (w * s, h * s);
    let keep = |view: f32, drawn: f32| ((view + drawn) / 2. - 48.).max(0.);
    (pan.0.clamp(-keep(bw, dw), keep(bw, dw)), pan.1.clamp(-keep(bh, dh), keep(bh, dh)))
}

pub fn image_rect(b: Bounds<Pixels>, w: f32, h: f32, fit: Fit, zoom: f32, pan: (f32, f32)) -> Bounds<Pixels> {
    let (bw, bh) = (f32::from(b.size.width), f32::from(b.size.height));
    let s = match fit {
        Fit::Cover => (bw / w).max(bh / h),
        Fit::Contain => (bw / w).min(bh / h),
    } * zoom;
    let (dw, dh) = (w * s, h * s);
    let pan = clamp_pan(b, w, h, fit, zoom, pan);
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
    reveal: bool,
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
        // Box corners land on whole device pixels: a fractional rect anti-aliases its halo and
        // stroke into each other and leaves see-through pixels where they should meet.
        let scale = window.scale_factor();
        let snap = |v: Pixels| px((f32::from(v) * scale).round() / scale);
        let to = |x: f32, y: f32| point(snap(rect.left() + px(x * sx)), snap(rect.top() + px(y * sy)));
        let halo = rgba(tokens::CLASS_HALO);
        let chip_text = rgba(tokens::CLASS_CHIP_TEXT);
        // Dimmed boxes first, highlighted last, so focus sits on top.
        let mut order: Vec<&DrawBox> = boxes.iter().collect();
        order.sort_by_key(|b| (b.highlight as u8, !b.dim as u8));
        // Label chips already placed, so later chips can step around them.
        let mut placed: Vec<Bounds<Pixels>> = Vec::new();
        // Each box's chip, by its index in `order`, once placed.
        let mut chips: Vec<Option<Chip>> = order.iter().map(|_| None).collect();
        // The boxes that matter (not dimmed): a dimmed box's chip keeps off them, so their strokes
        // never cross its text; with nowhere clear, a dimmed box goes unlabelled.
        // Every labelled box's outline, by index in `order`: a chip prefers a spot that crosses no
        // other box's stroke (in a cluster, a chip on a neighbour's edge reads as the neighbour's).
        let rects: Vec<Option<Bounds<Pixels>>> = order
            .iter()
            .map(|b| (!b.truth).then(|| Bounds::from_corners(to(b.x1, b.y1), to(b.x2, b.y2))))
            .collect();
        let strong: Vec<Bounds<Pixels>> = order
            .iter()
            .filter(|b| !b.dim && !b.truth)
            .map(|b| Bounds::from_corners(to(b.x1, b.y1), to(b.x2, b.y2)))
            .collect();
        let group = |b: &DrawBox| (b.highlight, b.dim);
        // Whether a `one_of` box has its chip yet.
        let mut one_done = false;
        // Two passes. First the chips are placed, the selected and matching boxes first, so they
        // claim the free spots before dimmed ones (a class filter must not leave its own boxes
        // unlabelled). Then everything paints a group at a time, dimmed first and the selected
        // last: a group's boxes, then its chips, so no stroke covers a chip of its own group and a
        // dimmed chip never sits over a matching box.
        for pass in 0..2 {
            let n = order.len();
            // Placement by importance: the selected box, then matches before dimmed ones, then
            // the more confident (predictions) or the larger (ground truth) first, so a cluster's
            // best detection gets the clear spot. Painting keeps `order`.
            let seq: Vec<usize> = if pass == 0 {
                let area = |b: &DrawBox| (b.x2 - b.x1) * (b.y2 - b.y1);
                let mut s: Vec<usize> = (0..n).collect();
                s.sort_by(|&x, &y| {
                    let (a, b) = (order[x], order[y]);
                    b.highlight
                        .cmp(&a.highlight)
                        .then(a.dim.cmp(&b.dim))
                        .then(b.score.unwrap_or(0.).total_cmp(&a.score.unwrap_or(0.)))
                        .then(area(b).total_cmp(&area(a)))
                });
                s
            } else {
                (0..n).collect()
            };
            for &i in &seq {
                let b = order[i];
                // Painting: the previous group is done, so its chips go on before this one starts.
                if pass == 1 && i > 0 && group(order[i - 1]) != group(b) {
                    for c in chips[..i].iter_mut().filter_map(Option::take) {
                        c.paint(window, cx);
                    }
                }
                let color = t.class(b.class);
                let a = if b.dim { 0.3 } else { 1.0 };
                let p1 = to(b.x1, b.y1);
                let p2 = to(b.x2, b.y2);
                let r = Bounds::from_corners(p1, p2);
                if pass == 1 {
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
                if pass == 0
                    && !b.truth
                    && labels
                    && !(b.one_of && one_done)
                    && !small
                    && !off_view
                    && (b.label.is_some() || b.score.is_some())
                {
                    // Small views (< 480 px): 14 px chips with 10 px text (design/README).
                    let compact = bounds.size.width < px(480.);
                    // Small views label sparingly at rest; hovered or focused (`reveal`), they label
                    // like full views, so no box is known by its colour alone.
                    let strict = compact && !reveal;
                    // small: label chips drawn on images (design/README: `xs`, `micro` on thumbnails).
                    let fs = if compact { 10. } else { ty::XS.size };
                    // Narrow views (thumbnails under 140 px) name the class in full and leave the
                    // score out: a word reads where a prefix ("tri") or a bare score can't (the
                    // inspector shows the scores).
                    let narrow = bounds.size.width < px(140.);
                    let name = b.label.clone().unwrap_or_default();
                    // The selected box's chip takes the selection colour, like its outline.
                    let (chip_bg, ink) = if b.highlight {
                        (t.accent_solid, t.accent_on)
                    } else if b.dim {
                        // Thumbnails: a dimmed box keeps only its faded outline, so the filtered
                        // class's chips carry the tile.
                        if strict {
                            continue;
                        }
                        // Larger views: muted but opaque (a translucent chip over a photo can't be
                        // read), its text a step back from the matches'.
                        let muted = crate::theme::over(t.bg_panel, alpha(color, 0.25));
                        (muted, crate::theme::alpha(t.readable_on(muted), 0.7))
                    } else {
                        (color, chip_text)
                    };
                    let score = b.score.filter(|_| !narrow).map(|s| format!("{s:.2}")).unwrap_or_default();
                    // Nothing left to say (a name-only chip in a narrow view): no empty chip.
                    if name.is_empty() && score.is_empty() {
                        continue;
                    }
                    let mut name_line = window.text_system().shape_line(
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
                    // A name too long for the view gives way, so the score is never clipped.
                    let mut name = name;
                    if !score.is_empty()
                        && !name.is_empty()
                        && name_line.width + px(5.) + score_line.width + px(10.) > bounds.size.width - px(4.)
                    {
                        name = Default::default();
                        name_line = window.text_system().shape_line(
                            name.clone(),
                            px(fs),
                            &[run("", SANS, FontWeight::MEDIUM, ink)],
                            None,
                        );
                    }
                    // A name-only chip wider than the view would run off it: none (the inspector
                    // and the detection list name the box in full). The selected box keeps its chip.
                    if score.is_empty()
                        && !b.highlight
                        && name_line.width + px(10.) > bounds.size.width - px(4.)
                    {
                        continue;
                    }
                    let gap = if score.is_empty() || name.is_empty() { px(0.) } else { px(5.) };
                    let cw = name_line.width + gap + score_line.width + px(10.);
                    let ch = px(if compact { 14. } else { 16. });
                    // Thumbnails label sparingly: a chip as big as the object it names is noise, and
                    // five chips on a 280 px tile outshout the image. Boxes at least two chips tall,
                    // at most four chips a tile; the rest keep their outline (the inspector, and the
                    // class key above the grid, name them).
                    // Measured on what's in view: a box cropped to a sliver at the tile's edge
                    // can't carry a chip (the chip would sit on someone else's pixels), and in any
                    // view a box less than half visible goes unlabelled unless it's selected.
                    let area = |x: &Bounds<Pixels>| f32::from(x.size.width) * f32::from(x.size.height);
                    let half_in = area(&visible) >= area(&r) * 0.5;
                    // A tile's one shared chip (`one_of`) needs only a chip's height plus 4 px of box:
                    // it's the tile's single name, so a 26 px match still carries it.
                    let tall_enough = visible.size.height >= if b.one_of { ch + px(4.) } else { ch * 2. };
                    if !b.highlight && (!half_in || strict && (!tall_enough || placed.len() >= 4)) {
                        continue;
                    }
                    // Preferred spots: above the top-left corner, inside it, below the bottom-left
                    // corner, inside the bottom-left. Take the first that is on the image and free.
                    // Inside the visible part of the box when the chip fits there, else inside the view.
                    let clamp_x = |x: Pixels| {
                        let lo = visible.left().max(bounds.left());
                        let hi = visible.right().min(bounds.right()) - cw;
                        if hi >= lo {
                            x.max(lo).min(hi)
                        } else {
                            x.min(bounds.right() - cw).max(bounds.left())
                        }
                    };
                    // Above the box (left, then right), then inside its top corners, then inside its
                    // bottom ones. Never below it: a chip under its box reads as the next one's.
                    // A right-anchored spot only when the chip fits the box's width: wider, it would
                    // hang off the box's left edge and over a neighbour.
                    let lx = r.left() - px(1.);
                    let rx = if cw <= r.size.width { r.right() - cw + px(1.) } else { lx };
                    let candidates = [
                        point(clamp_x(lx), r.top() - ch - px(1.)),
                        point(clamp_x(rx), r.top() - ch - px(1.)),
                        point(clamp_x(r.left() + px(1.)), r.top() + px(1.)),
                        point(clamp_x(rx - px(2.)), r.top() + px(1.)),
                        point(clamp_x(r.left() + px(1.)), r.bottom() - ch - px(1.)),
                        point(clamp_x(rx - px(2.)), r.bottom() - ch - px(1.)),
                    ];
                    let fits = |o: Point<Pixels>| {
                        o.y >= bounds.top().max(rect.top()) && o.y + ch <= bounds.bottom().min(rect.bottom())
                    };
                    let free = |c: &Bounds<Pixels>| {
                        !placed.iter().any(|p| p.dilate(px(2.)).intersects(c))
                            && !(b.dim && strong.iter().any(|s| s.dilate(px(2.)).intersects(c)))
                    };
                    // More spots down the inside of the box, one chip height (+2 px) apart, only
                    // while they stay inside it: a chip below its box would read as another's.
                    let inside = (1..4)
                        .map(|k| {
                            point(clamp_x(r.left() + px(1.)), r.top() + px(1.) + (ch + px(2.)) * k as f32)
                        })
                        .filter(|o| o.y + ch <= r.bottom() - px(1.));
                    // Whether a chip lies across another box's outline (touching its stroke band,
                    // not sitting wholly inside or outside it).
                    let crosses = |c: &Bounds<Pixels>| {
                        rects.iter().enumerate().any(|(j, o)| {
                            let Some(o) = o.filter(|_| j != i) else { return false };
                            let inner = o.dilate(px(-2.));
                            let within = c.left() >= inner.left()
                                && c.right() <= inner.right()
                                && c.top() >= inner.top()
                                && c.bottom() <= inner.bottom();
                            o.dilate(px(2.)).intersects(c) && !within
                        })
                    };
                    let spots = candidates.iter().copied().chain(inside);
                    let open = |o: &Point<Pixels>| fits(*o) && free(&Bounds::new(*o, size(cw, ch)));
                    // First a spot clear of every other outline (skipped on very dense images, where
                    // it costs a scan of every box per spot); in a cluster with none, any free spot.
                    let dense = order.len() > 256;
                    let clear = (!dense)
                        .then(|| spots.clone().find(|o| open(o) && !crosses(&Bounds::new(*o, size(cw, ch)))))
                        .flatten();
                    // A chip on a neighbour's outline can read as the neighbour's, at any size: such
                    // a box goes unlabelled (the list beside the image, the inspector and hover name
                    // it). The selected box always keeps its chip; very dense images skip the check.
                    // No clear spot: inside one of the box's own corners, flush with its stroke, so
                    // the chip shares a corner with its box and reads as its own even where it
                    // crosses a neighbour. The selected box and dense images take any free spot.
                    // Any of its four inside corners shares the box's stroke; the first free one.
                    let free_spot = clear.or_else(|| {
                        if dense || b.highlight {
                            spots.clone().find(open)
                        } else if !strict || b.one_of {
                            // Thumbnails at rest stay strict (too small for a shared corner to read),
                            // except for a tile's one shared chip: a match in a cluster still gets it.
                            candidates[2..].iter().copied().find(open)
                        } else {
                            None
                        }
                    });
                    // Nowhere free: skip this chip (the selected box's chip always shows), so two
                    // chips never paint over each other.
                    let Some(origin) = free_spot.or_else(|| {
                        b.highlight
                            .then(|| candidates.iter().copied().find(|o| fits(*o)).unwrap_or(candidates[2]))
                    }) else {
                        continue;
                    };
                    let chip = Bounds::new(origin, size(cw, ch));
                    placed.push(chip);
                    let (ty_, lh) =
                        if compact { (origin.y + px(1.), px(12.)) } else { (origin.y + px(1.5), px(13.)) };
                    one_done |= b.one_of;
                    chips[i] = Some(Chip {
                        bounds: chip,
                        bg: chip_bg,
                        name: name_line,
                        score: score_line,
                        gap,
                        ty: ty_,
                        lh,
                    });
                }
            }
        }
        // The last group's chips.
        for c in chips.iter_mut().filter_map(Option::take) {
            c.paint(window, cx);
        }
    });
}

/// A placed label chip, painted after its group's boxes.
struct Chip {
    bounds: Bounds<Pixels>,
    bg: Hsla,
    name: gpui::ShapedLine,
    score: gpui::ShapedLine,
    gap: Pixels,
    ty: Pixels,
    lh: Pixels,
}

impl Chip {
    fn paint(self, window: &mut Window, cx: &mut gpui::App) {
        // Dimmed chips fade their fill with the box; the text stays readable.
        window.paint_quad(fill(self.bounds, self.bg).corner_radii(px(tokens::radius::XS)));
        let x = self.bounds.origin.x + px(5.);
        let _ = self.name.paint(point(x, self.ty), self.lh, window, cx);
        let _ = self.score.paint(point(x + self.name.width + self.gap, self.ty), self.lh, window, cx);
    }
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
