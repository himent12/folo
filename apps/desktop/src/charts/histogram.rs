//! Histograms: vertical bars with labels under them.

use std::cell::Cell;
use std::rc::Rc;
use std::sync::Arc;

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Bounds, Hsla, InteractiveElement, IntoElement, MouseMoveEvent, ParentElement, Pixels,
    SharedString, StatefulInteractiveElement, Styled, canvas, div, fill, point, px, size,
};

use super::HoverFn;
use super::text::{paint_text, text_width};
use crate::theme::{Theme, alpha, ty};

/// Vertical bars for a histogram; `labels` sit under the bars (one per bin).
pub fn histogram(
    t: &Theme,
    id: impl Into<SharedString>,
    counts: Arc<Vec<u64>>,
    labels: Arc<Vec<String>>,
    color: Hsla,
    height: f32,
    hover: Option<f32>,
    on_hover: Option<HoverFn>,
) -> AnyElement {
    let t = *t;
    let cell: Rc<Cell<Option<Bounds<Pixels>>>> = Rc::new(Cell::new(None));
    let paint_cell = cell.clone();
    let leave = on_hover.clone();
    div()
        .id(folo_ui::ElementId::Name(id.into()))
        .w_full()
        .h(px(height))
        .when_some(on_hover, |d, f| {
            let cell = cell.clone();
            d.on_mouse_move(move |ev: &MouseMoveEvent, window, cx| {
                if let Some(b) = cell.get()
                    && b.size.width > px(0.)
                    && b.size.height > px(0.)
                {
                    let frac = ((ev.position.x - b.left()) / b.size.width).clamp(0.0, 0.9999);
                    f(b.contains(&ev.position).then_some(frac), window, cx);
                }
            })
        })
        .when_some(leave, |d, f| {
            d.on_hover(move |h, window, cx| {
                if !h {
                    f(None, window, cx)
                }
            })
        })
        .child(
            canvas(
                |_, _, _| (),
                move |b, _, window, cx| {
                    let n = counts.len().max(1);
                    let max = counts.iter().copied().max().unwrap_or(1).max(1) as f32;
                    let label_h = px(18.);
                    let plot = Bounds::new(
                        b.origin,
                        size(b.size.width.max(px(0.)), (b.size.height - label_h).max(px(0.))),
                    );
                    if plot.size.width <= px(0.) || plot.size.height <= px(0.) {
                        paint_cell.set(None);
                        return;
                    }
                    paint_cell.set(Some(plot));
                    let gap =
                        if n > 1 && plot.size.width >= px(5.) * n as f32 - px(4.) { px(4.) } else { px(0.) };
                    let bw = ((plot.size.width - gap * (n as f32 - 1.)) / n as f32).max(px(0.));
                    let hovered = hover
                        .filter(|f| f.is_finite() && (0.0..=1.0).contains(f))
                        .map(|f| ((f * n as f32).floor() as usize).min(n - 1));
                    window.paint_quad(fill(
                        Bounds::new(point(plot.left(), plot.bottom()), size(plot.size.width, px(1.))),
                        t.chart_axis,
                    ));
                    for (i, &c) in counts.iter().enumerate() {
                        let x = plot.left() + (bw + gap) * i as f32;
                        let h = (plot.size.height - px(16.)).max(px(0.)) * (c as f32 / max);
                        let h = if c > 0 { h.max(px(2.)).min(plot.size.height) } else { px(0.) };
                        let hot = hovered == Some(i);
                        let col = if hovered.is_some() && !hot { alpha(color, 0.45) } else { color };
                        window.paint_quad(
                            fill(Bounds::new(point(x, plot.bottom() - h), size(bw, h)), col).corner_radii(
                                folo_ui::Corners {
                                    top_left: px(2.),
                                    top_right: px(2.),
                                    bottom_left: px(0.),
                                    bottom_right: px(0.),
                                },
                            ),
                        );
                        if let Some(l) = labels.get(i) {
                            // small: category tick labels under the bars.
                            let fs = ty::XS.size;
                            let w = text_width(window, l, fs);
                            if w < bw + gap * 2. || i % 2 == 0 {
                                paint_text(
                                    window,
                                    cx,
                                    l,
                                    point(x + bw / 2. - w / 2., plot.bottom() + px(5.)),
                                    t.chart_label,
                                    fs,
                                    false,
                                );
                            }
                        }
                        if hot {
                            let s = crate::ui::int(c);
                            let w = text_width(window, &s, ty::SM.size);
                            let lx = (x + bw / 2. - w / 2.).clamp(b.left(), (b.right() - w).max(b.left()));
                            let ly = (plot.bottom() - h - px(15.)).max(b.top());
                            paint_text(window, cx, &s, point(lx, ly), t.text_primary, ty::SM.size, false);
                        }
                    }
                },
            )
            .size_full(),
        )
        .into_any_element()
}
