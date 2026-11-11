//! Scroll chrome: the thumb's geometry and painting, and the edge fades.

use folo_ui::{Bounds, Hsla, Pixels, ScrollHandle, Styled, fill, point, px, size};

use crate::theme::Theme;

/// Where a scroll container's thumb sits, from its handle (None when nothing overflows).
#[derive(Clone, Copy)]
pub struct ThumbGeo {
    pub top: Pixels,
    pub right: Pixels,
    pub h: Pixels,
    pub thumb: Pixels,
    pub y: Pixels,
    pub max: Pixels,
}

impl ThumbGeo {
    pub fn of(handle: &ScrollHandle) -> Option<Self> {
        let b = handle.bounds();
        let max = handle.max_offset().height;
        let h = b.size.height;
        if max <= px(0.5) || h <= px(8.) {
            return None;
        }
        let thumb = (h * (h / (h + max))).max(px(28.)).min(h);
        let off = (-handle.offset().y).clamp(px(0.), max);
        Some(Self { top: b.top(), right: b.right(), h, thumb, y: b.top() + (h - thumb) * (off / max), max })
    }
}

/// Paint the thumb (folo-ui draws none), so overflow is visible.
pub fn paint_thumb(
    t: &Theme,
    handle: &ScrollHandle,
    hot: bool,
    window: &mut folo_ui::Window,
    cx: &folo_ui::App,
) {
    let Some(g) = ThumbGeo::of(handle) else { return };
    // An overlay thumb: shown while dragged, while the pointer is over its area, and for a moment
    // after the content scrolls; otherwise the content has the space to itself. Movement is
    // recorded on every paint, whatever else shows the thumb.
    // The executor's clock, not the wall clock: a headless `--shot` runs on a fake clock and settles
    // like the real app (the thumb hides 800 ms after a scenario's scroll, before the capture).
    let scrolled = recently_scrolled(handle, cx.background_executor().now());
    let over = handle.bounds().contains(&window.mouse_position());
    if !(hot || over || scrolled) {
        return;
    }
    let bounds = Bounds::new(point(g.right - px(6.), g.y + px(2.)), size(px(4.), g.thumb - px(4.)));
    // Clipped away entirely (a scroll area below the page's fold, such as a live run's log following
    // its new lines): nothing to draw, and no frames to keep drawing it. A frame that brings it into
    // view paints it, and asks for more, while the scroll is still recent.
    if !bounds.intersects(&window.content_mask().bounds) {
        return;
    }
    // Shown only by a recent scroll: keep painting until that lapses, so it hides on time.
    if scrolled && !hot && !over {
        window.request_animation_frame();
    }
    // 3:1 against the surface in both themes (45 % read about 1.9:1).
    let color = crate::theme::alpha(t.text_tertiary, if hot { 1.0 } else { 0.8 });
    window.paint_quad(fill(bounds, color).corner_radii(px(2.)));
}

/// A scroll area's origin, its last vertical offset, and when that last changed.
type ScrollSeen = ((f32, f32), f32, std::time::Instant);

thread_local! {
    static SCROLLED: std::cell::RefCell<Vec<ScrollSeen>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Edge fades over a scroll area where more content lies past it (thumbs being hidden at
/// rest, the cue that there's more).
pub fn scroll_edges(
    handle: &ScrollHandle,
    bg: Hsla,
    inset: f32,
    radius: f32,
) -> impl Iterator<Item = folo_ui::Canvas<()>> {
    // Read at paint time, after this frame's layout set the scroll extent: a fade decided while
    // building the frame used the last frame's extent, so a window's first frames (and a still
    // capture) showed no fade over content below the fold.
    let h = handle.clone();
    let painter = folo_ui::canvas(
        |_, _, _| (),
        move |b, _, window, _| {
            let y = h.offset().y;
            let above = y < px(-1.);
            let below = h.max_offset().height + y > px(1.);
            // 24 px: the gradient holds its full colour for the outer 6 px (its first stop at
            // 25%), so a row cut at the edge leaves no sliver of glyphs, then ramps out. It
            // paints its own rounded corners.
            let (inset, r) = (px(inset), px(radius));
            let w = b.size.width - inset * 2.;
            let fade = |top: bool| {
                let at = if top { b.top() + inset } else { b.bottom() - inset - px(24.) };
                let corners = if top {
                    folo_ui::Corners { top_left: r, top_right: r, ..Default::default() }
                } else {
                    folo_ui::Corners { bottom_left: r, bottom_right: r, ..Default::default() }
                };
                folo_ui::quad(
                    Bounds::new(point(b.left() + inset, at), size(w, px(24.))),
                    corners,
                    folo_ui::linear_gradient(
                        if top { 180. } else { 0. },
                        folo_ui::linear_color_stop(bg, 0.25),
                        folo_ui::linear_color_stop(crate::theme::alpha(bg, 0.), 1.),
                    ),
                    px(0.),
                    folo_ui::transparent_black(),
                    folo_ui::BorderStyle::default(),
                )
            };
            if above {
                window.paint_quad(fade(true));
            }
            if below {
                window.paint_quad(fade(false));
            }
        },
    );
    std::iter::once(painter.absolute().top_0().left_0().size_full())
}

/// Whether `handle`'s content moved in the 800 ms before `now`.
fn recently_scrolled(handle: &ScrollHandle, now: std::time::Instant) -> bool {
    let b = handle.bounds();
    let key = (f32::from(b.origin.x), f32::from(b.origin.y));
    let y = f32::from(handle.offset().y);
    SCROLLED.with(|s| {
        let mut s = s.borrow_mut();
        match s.iter_mut().find(|(k, _, _)| *k == key) {
            // A jump back to the top is a page change resetting its scroll, not the user scrolling.
            Some(e) if (e.1 - y).abs() > 0.5 && y.abs() < 0.5 => {
                *e = (key, y, now - std::time::Duration::from_secs(1));
                false
            }
            Some(e) if (e.1 - y).abs() > 0.5 => {
                *e = (key, y, now);
                true
            }
            Some(e) => now.duration_since(e.2) < std::time::Duration::from_millis(800),
            None => {
                // First sight: not a scroll.
                s.push((key, y, now - std::time::Duration::from_secs(1)));
                if s.len() > 64 {
                    s.remove(0);
                }
                false
            }
        }
    })
}
