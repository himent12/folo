//! Sparklines: tiny axis-less trend lines.

use std::sync::Arc;

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Bounds, Hsla, IntoElement, PathBuilder, Pixels, Point, Styled, canvas, fill,
    linear_color_stop, linear_gradient, point, px, size,
};

use super::line::stroke_path;
use crate::theme::{Theme, alpha};

/// A tiny axis-less trend line with an optional area and live end dot.
pub fn sparkline(t: &Theme, values: Arc<Vec<f32>>, color: Hsla, height: f32, live: bool) -> AnyElement {
    sparkline_in(t, values, color, height, live, None)
}

/// A sparkline over a fixed value range (e.g. 0..1 for utilisation), so small wobbles stay small.
pub fn sparkline_in(
    t: &Theme,
    values: Arc<Vec<f32>>,
    color: Hsla,
    height: f32,
    live: bool,
    range: Option<(f32, f32)>,
) -> AnyElement {
    let t = *t;
    canvas(
        |_, _, _| (),
        move |b, _, window, _| {
            if values.len() < 2 {
                return;
            }
            let (lo, hi) = range.unwrap_or_else(|| {
                values.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &v| (lo.min(v), hi.max(v)))
            });
            let span = (hi - lo).max(1e-6);
            let inset = px(3.);
            let w = b.size.width - inset * 2.;
            let h = b.size.height - inset * 2.;
            let n = values.len();
            let pts: Vec<Point<Pixels>> = values
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    point(
                        b.left() + inset + w * (i as f32 / (n - 1) as f32),
                        b.top() + inset + h * (1.0 - (v - lo) / span),
                    )
                })
                .collect();
            let mut a = PathBuilder::fill();
            a.move_to(point(pts[0].x, b.bottom()));
            for p in &pts {
                a.line_to(*p);
            }
            a.line_to(point(pts[n - 1].x, b.bottom()));
            a.close();
            if let Ok(path) = a.build() {
                window.paint_path(
                    path,
                    linear_gradient(
                        180.,
                        linear_color_stop(alpha(color, t.chart_area_opacity * 1.4), 0.),
                        linear_color_stop(alpha(color, 0.), 1.),
                    ),
                );
            }
            stroke_path(window, &pts, 1.25, color, &[]);
            if live {
                let c = pts[n - 1];
                window.paint_quad(
                    fill(Bounds::centered_at(c, size(px(4.), px(4.))), color).corner_radii(px(2.)),
                );
            }
        },
    )
    .w_full()
    .map(|d| if height > 0. { d.h(px(height)) } else { d.h_full() })
    .into_any_element()
}
