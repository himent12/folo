//! Images with detection boxes, painted on a canvas.
//!
//! Boxes follow design/README.md: a 2 px stroke in the class colour wrapped by a 1 px halo so it
//! reads on any photo, square corners (boxes are measurements), and a label chip on the box's
//! top-left corner outside the box, moved inside when it would leave the image.

use std::sync::{Arc, LazyLock};

use folo_ui::{
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
    if !w.is_finite()
        || !h.is_finite()
        || w <= 0.
        || h <= 0.
        || !bw.is_finite()
        || !bh.is_finite()
        || bw <= 0.
        || bh <= 0.
        || !zoom.is_finite()
        || zoom <= 0.
    {
        return (0., 0.);
    }
    let s = match fit {
        Fit::Cover => (bw / w).max(bh / h),
        Fit::Contain => (bw / w).min(bh / h),
    } * zoom;
    let (dw, dh) = (w * s, h * s);
    let keep = |view: f32, drawn: f32| ((view + drawn) / 2. - 48.).max(0.);
    let px_ = if pan.0.is_finite() { pan.0 } else { 0. };
    let py_ = if pan.1.is_finite() { pan.1 } else { 0. };
    (px_.clamp(-keep(bw, dw), keep(bw, dw)), py_.clamp(-keep(bh, dh), keep(bh, dh)))
}

pub fn image_rect(b: Bounds<Pixels>, w: f32, h: f32, fit: Fit, zoom: f32, pan: (f32, f32)) -> Bounds<Pixels> {
    let (bw, bh) = (f32::from(b.size.width).max(0.), f32::from(b.size.height).max(0.));
    if !w.is_finite() || !h.is_finite() || w <= 0. || h <= 0. || !zoom.is_finite() || zoom <= 0. {
        return Bounds::new(b.origin, size(px(0.), px(0.)));
    }
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

static TNUM_FEATURES: LazyLock<folo_ui::FontFeatures> =
    LazyLock::new(|| folo_ui::FontFeatures(Arc::new(vec![("tnum".into(), 1)])));

fn run(text: &str, family: &'static str, weight: FontWeight, color: Hsla) -> TextRun {
    TextRun {
        len: text.len(),
        font: folo_ui::Font { weight, features: TNUM_FEATURES.clone(), ..folo_ui::font(family) },
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
    // Match `Window::paint_image`'s device-pixel rounding (origin floored, size ceiled) so the
    // overlay clip mask lines up with the painted photo edges even at fractional pans/scales.
    let scale = window.scale_factor();
    let raw_rect = image_rect(bounds, w.max(1.), h.max(1.), fit, zoom, pan);
    let rect = Bounds::new(
        point(
            px((f32::from(raw_rect.origin.x) * scale).floor() / scale),
            px((f32::from(raw_rect.origin.y) * scale).floor() / scale),
        ),
        size(
            px((f32::from(raw_rect.size.width) * scale).ceil() / scale),
            px((f32::from(raw_rect.size.height) * scale).ceil() / scale),
        ),
    );
    let clip = bounds.intersect(&rect);
    window.with_content_mask(Some(folo_ui::ContentMask { bounds }), |window| {
        match img {
            Some(img) => {
                let r = if fit == Fit::Cover { corner } else { 0. };
                let _ = window.paint_image(rect, Corners::all(px(r)), img.clone(), 0, false);
            }
            None => window.paint_quad(fill(rect, t.bg_raised)),
        }
        if clip.size.width <= px(0.) || clip.size.height <= px(0.) {
            return;
        }
        // Clip boxes and label chips to the displayed photo (`bounds ∩ rect`) so unclipped
        // benchmark detections never paint into letterbox padding.
        window.with_content_mask(Some(folo_ui::ContentMask { bounds: clip }), |window| {
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
                    // Cropped out of view (Fill) or extending past photo edges: no chip in the padding.
                    let visible = r.intersect(&clip);
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
                        let score =
                            b.score.filter(|_| !narrow).map(|s| format!("{s:.2}")).unwrap_or_default();
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
                            && name_line.width + px(5.) + score_line.width + px(10.)
                                > clip.size.width - px(4.)
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
                            && name_line.width + px(10.) > clip.size.width - px(4.)
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
                            let lo = visible.left().max(clip.left());
                            let hi = visible.right().min(clip.right()) - cw;
                            if hi >= lo {
                                x.max(lo).min(hi)
                            } else {
                                x.min(clip.right() - cw).max(clip.left())
                            }
                        };
                        // Above the box (left, then right), then inside its top corners, then inside its
                        // bottom ones. Never below it: a chip under its box reads as the next one's.
                        // A right-anchored spot only when the chip fits the box's width: wider, it would
                        // hang off the box's left edge and over a neighbour.
                        let vt = r.top().max(clip.top());
                        let vb = r.bottom().min(clip.bottom());
                        let lx = visible.left() - px(1.);
                        let rx = if cw <= visible.size.width { visible.right() - cw + px(1.) } else { lx };
                        let candidates = [
                            point(clamp_x(lx), r.top() - ch - px(1.)),
                            point(clamp_x(rx), r.top() - ch - px(1.)),
                            point(clamp_x(visible.left() + px(1.)), vt + px(1.)),
                            point(clamp_x(rx - px(2.)), vt + px(1.)),
                            point(clamp_x(visible.left() + px(1.)), vb - ch - px(1.)),
                            point(clamp_x(rx - px(2.)), vb - ch - px(1.)),
                        ];
                        let fits = |o: Point<Pixels>| o.y >= clip.top() && o.y + ch <= clip.bottom();
                        let free = |c: &Bounds<Pixels>| {
                            !placed.iter().any(|p| p.dilate(px(2.)).intersects(c))
                                && !(b.dim && strong.iter().any(|s| s.dilate(px(2.)).intersects(c)))
                        };
                        // More spots down the inside of the box, one chip height (+2 px) apart, only
                        // while they stay inside it: a chip below its box would read as another's.
                        let inside = (1..4)
                            .map(|k| {
                                point(
                                    clamp_x(visible.left() + px(1.)),
                                    vt + px(1.) + (ch + px(2.)) * k as f32,
                                )
                            })
                            .filter(|o| o.y + ch <= vb - px(1.));
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
                            .then(|| {
                                spots.clone().find(|o| open(o) && !crosses(&Bounds::new(*o, size(cw, ch))))
                            })
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
                            b.highlight.then(|| {
                                candidates.iter().copied().find(|o| fits(*o)).unwrap_or(candidates[2])
                            })
                        }) else {
                            continue;
                        };
                        let chip = Bounds::new(origin, size(cw, ch));
                        placed.push(chip);
                        let (ty_, lh) = if compact {
                            (origin.y + px(1.), px(12.))
                        } else {
                            (origin.y + px(1.5), px(13.))
                        };
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
    });
}

/// A placed label chip, painted after its group's boxes.
struct Chip {
    bounds: Bounds<Pixels>,
    bg: Hsla,
    name: folo_ui::ShapedLine,
    score: folo_ui::ShapedLine,
    gap: Pixels,
    ty: Pixels,
    lh: Pixels,
}

impl Chip {
    fn paint(self, window: &mut Window, cx: &mut folo_ui::App) {
        // Dimmed chips fade their fill with the box; the text stays readable.
        window.paint_quad(fill(self.bounds, self.bg).corner_radii(px(tokens::radius::XS)));
        let x = self.bounds.origin.x + px(5.);
        let _ = self.name.paint(point(x, self.ty), self.lh, window, cx);
        let _ = self.score.paint(point(x + self.name.width + self.gap, self.ty), self.lh, window, cx);
    }
}

fn rgba(v: u32) -> Hsla {
    folo_ui::rgba(v).into()
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
    if !new_zoom.is_finite() || new_zoom <= 1.0 {
        return (0., 0.);
    }
    let rect = image_rect(bounds, w, h, Fit::Contain, zoom, pan);
    let (rw, rh) = (f32::from(rect.size.width), f32::from(rect.size.height));
    if rw <= 0. || rh <= 0. {
        return (0., 0.);
    }
    let fx = f32::from(cursor.x - rect.left()) / rw;
    let fy = f32::from(cursor.y - rect.top()) / rh;
    let fresh = image_rect(bounds, w, h, Fit::Contain, new_zoom, (0., 0.));
    let want_x = f32::from(cursor.x) - fx * f32::from(fresh.size.width);
    let want_y = f32::from(cursor.y) - fy * f32::from(fresh.size.height);
    let next = (want_x - f32::from(fresh.left()), want_y - f32::from(fresh.top()));
    clamp_pan(bounds, w, h, Fit::Contain, new_zoom, next)
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
    let (rw, rh) = (f32::from(rect.size.width), f32::from(rect.size.height));
    if rw <= 0. || rh <= 0. {
        return (-1., -1.);
    }
    let x = f32::from(p.x - rect.left()) / rw * w;
    let y = f32::from(p.y - rect.top()) / rh * h;
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

    #[test]
    fn degenerate_bounds_and_dims_never_produce_nan() {
        let valid_b = Bounds::new(point(px(10.), px(20.)), size(px(400.), px(300.)));
        let zero_b = Bounds::new(point(px(10.), px(20.)), size(px(0.), px(300.)));
        let neg_b = Bounds::new(point(px(10.), px(20.)), size(px(-10.), px(-20.)));

        for (b, dims, zoom, pan) in [
            (valid_b, (0., 480.), 2.0, (10., 10.)),
            (valid_b, (640., 0.), 2.0, (10., 10.)),
            (valid_b, (f32::NAN, 480.), 2.0, (10., 10.)),
            (zero_b, (640., 480.), 2.0, (10., 10.)),
            (neg_b, (640., 480.), 2.0, (10., 10.)),
            (valid_b, (640., 480.), 0.0, (10., 10.)),
            (valid_b, (640., 480.), f32::NAN, (f32::NAN, f32::INFINITY)),
        ] {
            let p = clamp_pan(b, dims.0, dims.1, Fit::Contain, zoom, pan);
            assert!(p.0.is_finite() && p.1.is_finite(), "clamp_pan produced non-finite: {p:?}");
            let r = image_rect(b, dims.0, dims.1, Fit::Contain, zoom, pan);
            assert!(
                f32::from(r.origin.x).is_finite()
                    && f32::from(r.origin.y).is_finite()
                    && f32::from(r.size.width).is_finite()
                    && f32::from(r.size.height).is_finite(),
                "image_rect produced non-finite: {r:?}"
            );
            let z = zoom_about(b, dims, zoom, pan, point(px(50.), px(50.)), 2.5);
            assert!(z.0.is_finite() && z.1.is_finite(), "zoom_about produced non-finite: {z:?}");
            let m = to_image(b, dims, Fit::Contain, zoom, pan, point(px(50.), px(50.)));
            assert!(m.0.is_finite() && m.1.is_finite(), "to_image produced non-finite: {m:?}");
        }
        assert_eq!(
            to_image(zero_b, (640., 480.), Fit::Contain, 1.0, (0., 0.), point(px(10.), px(20.))),
            (-1., -1.)
        );
        assert_eq!(clamp_pan(valid_b, 640., 480., Fit::Contain, 2.0, (f32::NAN, f32::INFINITY)), (0., 0.));
    }

    #[test]
    fn unclipped_detections_never_paint_into_letterbox_padding() {
        use folo_ui::{
            AppContext, Context, HeadlessApp, HeadlessOptions, IntoElement, ParentElement, Render,
            RenderImage, Styled, TextBackend, WindowBounds, WindowOptions, canvas, div, rgb,
        };

        // Solid blue 100x100 photo (BGRA premultiplied: B=0xC0, G=0x40, R=0x20, A=0xFF -> 0x002040C0).
        let mut bgra = Vec::with_capacity(100 * 100 * 4);
        for _ in 0..(100 * 100) {
            bgra.extend_from_slice(&[0xC0, 0x40, 0x20, 0xFF]);
        }
        let img = Arc::new(RenderImage::new(vec![image::Frame::new(
            image::RgbaImage::from_raw(100, 100, bgra).expect("valid photo buffer"),
        )]));

        struct View(Arc<RenderImage>);
        impl Render for View {
            fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
                let t = crate::theme::theme(cx);
                let img = self.0.clone();
                let boxes = vec![
                    // Partially outside left edge with `half_in == true`: its label chip + border paint.
                    DrawBox {
                        label: Some("circle".into()),
                        score: Some(0.95),
                        ..DrawBox::new(0, -10., 5., 70., 80.)
                    },
                    // Highlighted oversized detection crossing all four edges.
                    DrawBox {
                        label: Some("wide".into()),
                        score: Some(0.88),
                        highlight: true,
                        ..DrawBox::new(1, -40., -10., 140., 110.)
                    },
                    DrawBox { truth: true, ..DrawBox::new(2, -20., 10., 120., 90.) },
                ];
                div().size_full().bg(rgb(0x101010)).child(
                    canvas(
                        |_, _, _| (),
                        move |b, _, window, cx| {
                            // Fractional pan (0.75, 0.0) exercises device-pixel alignment between
                            // `paint_image` (origin floored) and the overlay clip mask.
                            paint_image_with_boxes(
                                &t,
                                window,
                                cx,
                                b,
                                Some(&img),
                                (100., 100.),
                                &boxes,
                                Fit::Contain,
                                1.0,
                                (0.75, 0.0),
                                true,
                                true,
                                0.,
                            );
                        },
                    )
                    .size_full(),
                )
            }
        }

        let window_size = size(px(400.), px(200.));
        let mut app = HeadlessApp::new(HeadlessOptions {
            size: window_size,
            scale_factor: 1.0,
            text: TextBackend::Native,
        });
        let window = app.update(|cx: &mut App| {
            cx.text_system()
                .add_fonts(crate::FONTS.iter().map(|f| std::borrow::Cow::Borrowed(*f)).collect())
                .expect("bundled fonts load");
            cx.set_global(Theme::new(crate::theme::Mode::Dark));
            let options = WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(Bounds::new(point(px(0.), px(0.)), window_size))),
                ..Default::default()
            };
            cx.open_window(options, |_, cx| cx.new(|_| View(img))).expect("open window")
        });
        let frame = app.frame(window.into());
        // Square 100x100 image in 400x200 Fit::Contain occupies x=100..300; padding is x=0..100 and x=300..400.
        for y in 0..200 {
            for x in (0..100).chain(300..400) {
                assert_eq!(
                    frame.pixel(x, y),
                    0x0010_1010,
                    "unclipped box/chip leaked into letterbox padding at ({x}, {y})"
                );
            }
        }
        // Inside the photo area, both the blue photo (0x0020_40C0) AND distinct box/chip overlay pixels painted.
        let mut overlay_pixels = 0usize;
        for y in 0..200 {
            for x in 100..300 {
                let p = frame.pixel(x, y);
                if p != 0x0010_1010 && p != 0x0020_40C0 {
                    overlay_pixels += 1;
                }
            }
        }
        assert!(overlay_pixels > 200, "box strokes and label chips painted inside photo: {overlay_pixels}");
    }
}
