//! Lines named at their ends (Studio's direct labels, `web/src/lib/charts/endLabels.ts`): a column
//! of labels, each in its own 16 px slot at its line's last height where it can be, so two lines
//! that end together get neighbouring labels that never touch. A column taller than the plot keeps
//! the labels that come first and folds the rest into one "+N more".

use folo_ui::{App, Bounds, Pixels, Point, Window, point, px};

use super::spec::{LineChart, Series};
use super::text::{paint_text, text_width};
use crate::theme::{Theme, ty};

/// A label's slot in the column, px: its whole 16 px line.
pub const SLOT: f32 = 16.;

/// The narrowest plot that names its lines: below it the legend carries the names alone.
const MIN_PLOT: f32 = 240.;

/// Whether a chart names its lines at their ends, and the gutter right of the plot that takes (px):
/// none when the lines end short of the plot's right edge with room beside them (a live run's
/// planned epochs), else a column as wide as the longest name, if the plot keeps `MIN_PLOT`.
pub(super) fn end_gutter(
    spec: &LineChart,
    window: &mut Window,
    x0: f32,
    x1: f32,
    plot_w: Pixels,
) -> (bool, Pixels) {
    let named: Vec<&Series> =
        spec.series.iter().filter(|s| s.end_label.is_some() && !s.points.is_empty()).collect();
    if !spec.end_labels || named.is_empty() {
        return (false, px(0.));
    }
    let w = named.iter().map(|s| text_width(window, &s.name, ty::SM.size)).fold(px(0.), |a, b| a.max(b));
    let now = named.iter().filter_map(|s| s.points.last()).map(|p| p.0).fold(f32::NEG_INFINITY, f32::max);
    let beside = (1. - ((now - x0) / (x1 - x0)).clamp(0., 1.)) * plot_w;
    if beside >= w + px(16.) {
        return (true, px(0.));
    }
    let gutter = w + px(10.);
    if plot_w - gutter < px(MIN_PLOT) { (false, px(0.)) } else { (true, gutter) }
}

/// Paints the names of the lines in `ends` that carry one: in the gutter right of the plot, or
/// (`in_gutter` false) right beside the furthest line end, each at its line's last height.
pub(super) fn paint_end_names(
    window: &mut Window,
    cx: &mut App,
    t: &Theme,
    ends: &[(Point<Pixels>, &Series)],
    plot: Bounds<Pixels>,
    in_gutter: bool,
) {
    let named: Vec<&(Point<Pixels>, &Series)> = ends.iter().filter(|(_, s)| s.end_label.is_some()).collect();
    let ys: Vec<f32> = named.iter().map(|(c, _)| f32::from(c.y)).collect();
    let x = if in_gutter {
        plot.right() + px(6.)
    } else {
        named.iter().map(|(c, _)| c.x).fold(plot.left(), |a, b| a.max(b)) + px(9.)
    };
    let (at, more) = stack(&ys, f32::from(plot.top()), f32::from(plot.bottom()));
    for ((_, s), y) in named.iter().zip(at) {
        if let (Some(y), Some(ink)) = (y, s.end_label) {
            paint_text(window, cx, &s.name, point(x, px(y - SLOT / 2.)), ink, ty::SM.size, false);
        }
    }
    if more > 0 && plot.size.height >= px(SLOT) {
        let y = f32::from(plot.bottom()) - SLOT;
        paint_text(
            window,
            cx,
            &format!("+{more} more"),
            point(x, px(y)),
            t.text_tertiary,
            ty::SM.size,
            false,
        );
    }
}

/// Where each label's centre goes (same order as `ys`), given the lines' last heights `ys` (px,
/// down) and the column's span `top..bottom`: each as near its line's end as the slots allow, every
/// two at least one `SLOT` apart, all inside the span. `None` for a label folded away: it is told in
/// the column's last slot as "+N more" (`more`).
pub fn stack(ys: &[f32], top: f32, bottom: f32) -> (Vec<Option<f32>>, usize) {
    let cap =
        if bottom - top < SLOT { 0 } else { ((bottom - top - SLOT) / SLOT + 1e-4).floor() as usize + 1 };
    let mut order: Vec<usize> = (0..ys.len()).collect();
    order.sort_by(|&a, &b| ys[a].total_cmp(&ys[b]));
    // Too many for the column: the first `cap - 1` from the top keep their labels, the rest fold
    // into the last slot.
    let fold = ys.len() > cap;
    let kept = if fold { cap.saturating_sub(1) } else { ys.len() };
    let lo = top + SLOT / 2.;
    let hi = bottom - SLOT / 2.;
    let mut at: Vec<f32> = order[..kept].iter().map(|&i| ys[i].clamp(lo, hi)).collect();
    // Down from the top: each a slot under the one above it.
    for k in 1..at.len() {
        at[k] = at[k].max(at[k - 1] + SLOT);
    }
    // Up from the bottom: a column pushed past the span moves back up, a slot apart.
    let last = if fold { hi - SLOT } else { hi };
    if let Some(l) = at.last_mut() {
        *l = l.min(last);
    }
    for k in (0..at.len().saturating_sub(1)).rev() {
        at[k] = at[k].min(at[k + 1] - SLOT);
    }
    let mut out = vec![None; ys.len()];
    for (k, &i) in order[..kept].iter().enumerate() {
        out[i] = Some(at[k]);
    }
    (out, if fold { ys.len() - kept } else { 0 })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_label_sits_at_its_lines_end_when_nothing_is_near() {
        let (at, more) = stack(&[40., 120.], 0., 200.);
        assert_eq!(at, vec![Some(40.), Some(120.)]);
        assert_eq!(more, 0);
    }

    #[test]
    fn two_lines_ending_together_get_neighbouring_slots_that_never_touch() {
        // Loss's box and dfl end a pixel apart.
        let (at, _) = stack(&[100., 101., 30.], 0., 200.);
        let (a, b) = (at[0].unwrap(), at[1].unwrap());
        assert!((b - a).abs() >= SLOT - 1e-4, "{a} {b}");
        assert!(a <= 101. && b >= 100.);
        assert_eq!(at[2], Some(30.));
    }

    #[test]
    fn labels_stay_inside_the_plot() {
        let (at, _) = stack(&[195., 196., 197.], 0., 200.);
        for y in at.iter().flatten() {
            assert!(*y >= SLOT / 2. && *y <= 200. - SLOT / 2., "{y}");
        }
        let (at, _) = stack(&[-20.], 0., 200.);
        assert_eq!(at, vec![Some(SLOT / 2.)]);
    }

    #[test]
    fn a_column_too_tall_folds_the_rest_into_one_more() {
        // Room for three slots: two labels and "+2 more".
        let (at, more) = stack(&[10., 20., 30., 40.], 0., 48.);
        assert_eq!(more, 2);
        assert_eq!(at.iter().flatten().count(), 2);
        assert_eq!(at[2], None);
        assert_eq!(at[3], None);
        let (at, more) = stack(&[10.], 0., 8.);
        assert_eq!((at, more), (vec![None], 1));
    }
}
