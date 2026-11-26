//! The hover cursor: the values under it, where it snaps and its chip's text.

use super::smooth::smoothed;
use super::spec::{LineChart, Series};

/// The y at `x` if the series has a point within 6 % of `span` of it (the cursor-dot rule), else
/// None: past a curve's end there is no value to show.
pub fn value_near(points: &[(f32, f32)], x: f32, span: f32) -> Option<f32> {
    if !x.is_finite() || !span.is_finite() || span < 0.0 {
        return None;
    }
    let (px_, py) = points[nearest(points, x)?];
    (px_.is_finite() && py.is_finite() && (px_ - x).abs() <= span * 0.06).then_some(py)
}

/// The cursor chip's text, named ("epoch 36") so a linked cursor's unit is explicit.
pub(super) fn chip_label(spec: &LineChart, x: f32) -> String {
    let v = cursor_value(spec, x);
    if spec.x_name.is_empty() { v } else { format!("{} {v}", spec.x_name) }
}

/// The cursor's x as text: `cursor_fmt`, else the tick format.
pub(super) fn cursor_value(spec: &LineChart, x: f32) -> String {
    spec.cursor_fmt.as_ref().map_or_else(|| (spec.x_fmt)(x), |f| f(x))
}

/// Index of the point nearest to `x` in a sorted series.
/// The value a legend shows: at its chart's cursor when there is one, else the latest point.
pub fn value_at(points: &[(f32, f32)], hover: Option<f32>, x0: f32, x1: f32) -> Option<f32> {
    match hover {
        Some(f) if f.is_finite() && x0.is_finite() && x1.is_finite() => {
            nearest(points, x0 + (x1 - x0) * f).map(|i| points[i].1)
        }
        Some(_) => None,
        None => points.iter().rev().find(|p| p.0.is_finite() && p.1.is_finite()).map(|p| p.1),
    }
}

/// The x the cursor snaps to at `hx`: the first series point in `[x0, x1]` within 6 % of the
/// range (the same rule the cursor painter uses), or None when the cursor is past the data.
pub fn cursor_x(series: &[Series], hx: f32, x0: f32, x1: f32) -> Option<f32> {
    let span = x1 - x0;
    if !hx.is_finite() || !x0.is_finite() || !x1.is_finite() || !span.is_finite() || span < 0.0 {
        return None;
    }
    series.iter().find_map(|s| {
        let data = if s.smooth > 0.0 { smoothed(&s.points, s.smooth) } else { s.points.clone() };
        let (x, y) = data[nearest_in_range(&data, hx, x0, x1)?];
        (x.is_finite() && y.is_finite() && ((x - hx).abs() <= span * 0.06 || s.points.len() <= 1))
            .then_some(x)
    })
}

/// Index of the finite point nearest to `x` whose x-coordinate lies within `[x0, x1]`.
pub fn nearest_in_range(points: &[(f32, f32)], x: f32, x0: f32, x1: f32) -> Option<usize> {
    if points.is_empty() || !x.is_finite() || !x0.is_finite() || !x1.is_finite() || x0 > x1 {
        return None;
    }
    let lo = points.partition_point(|p| p.0 < x0);
    let hi = points.partition_point(|p| p.0 <= x1);
    if lo < hi
        && let Some(rel) = nearest(&points[lo..hi], x)
    {
        let p = points[lo + rel];
        if p.0.is_finite() && p.1.is_finite() && p.0 >= x0 && p.0 <= x1 {
            return Some(lo + rel);
        }
    }
    points
        .iter()
        .enumerate()
        .filter(|(_, p)| p.0.is_finite() && p.1.is_finite() && p.0 >= x0 && p.0 <= x1)
        .min_by(|(_, a), (_, b)| (a.0 - x).abs().total_cmp(&(b.0 - x).abs()))
        .map(|(idx, _)| idx)
}

pub fn nearest(points: &[(f32, f32)], x: f32) -> Option<usize> {
    if points.is_empty() || !x.is_finite() {
        return None;
    }
    let i = points.partition_point(|p| p.0 < x);
    let idx = if i == 0 {
        0
    } else if i >= points.len() {
        points.len() - 1
    } else if (points[i].0 - x).abs() < (x - points[i - 1].0).abs() {
        i
    } else {
        i - 1
    };
    if points[idx].0.is_finite() && points[idx].1.is_finite() {
        return Some(idx);
    }
    points
        .iter()
        .enumerate()
        .filter(|(_, p)| p.0.is_finite() && p.1.is_finite())
        .min_by(|(_, a), (_, b)| (a.0 - x).abs().total_cmp(&(b.0 - x).abs()))
        .map(|(idx, _)| idx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn cursor_past_the_data_has_no_chip() {
        let pts: Vec<(f32, f32)> = (0..=40).map(|i| (i as f32, 0.5)).collect();
        let s = vec![Series::new("a", folo_ui::hsla(0., 0., 0., 1.), Arc::new(pts))];
        // Epoch 100 of 120 with data through 40: no snap, so no tick is hidden.
        assert_eq!(cursor_x(&s, 100., 0., 120.), None);
        assert_eq!(cursor_x(&s, 39.6, 0., 120.), Some(40.));
    }

    #[test]
    fn cursor_respects_visible_domain_and_ignores_closer_out_of_range_points() {
        // Point at 9.9 is closer to 10.0 than 10.2, but outside [10.0, 20.0]: cursor must snap to
        // the in-range point 10.2 rather than rejecting the series.
        let pts = vec![(9.9, 1.0), (10.2, 2.0), (20.1, 3.0)];
        assert_eq!(nearest_in_range(&pts, 10.0, 10.0, 20.0), Some(1));
        assert_eq!(nearest_in_range(&pts, 20.0, 10.0, 20.0), Some(1));
        let s = vec![Series::new("a", folo_ui::hsla(0., 0., 0., 1.), Arc::new(pts))];
        assert_eq!(cursor_x(&s, 10.0, 10.0, 20.0), Some(10.2));
        // A series with only out-of-range points must not produce a chip span.
        let outside = vec![Series::new("b", folo_ui::hsla(0., 0., 0., 1.), Arc::new(vec![(5.0, 1.0)]))];
        assert_eq!(cursor_x(&outside, 10.0, 10.0, 20.0), None);
    }

    #[test]
    fn no_value_past_a_curves_end() {
        let pts = [(0.0, 1.0), (0.2, 0.9), (0.4, 0.5)];
        assert_eq!(value_near(&pts, 0.41, 1.), Some(0.5));
        assert_eq!(value_near(&pts, 0.9, 1.), None);
    }

    #[test]
    fn nearest_point() {
        let pts = vec![(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)];
        assert_eq!(nearest(&pts, 1.4), Some(1));
        assert_eq!(nearest(&pts, 1.6), Some(2));
        assert_eq!(nearest(&pts, -3.0), Some(0));
        assert_eq!(nearest(&[], 1.0), None);
        assert_eq!(nearest(&pts, f32::NAN), None);
        assert_eq!(value_near(&pts, f32::NAN, 2.0), None);
        assert_eq!(value_at(&pts, Some(f32::NAN), 0.0, 2.0), None);
        let with_nan = vec![(0.0, 1.0), (1.0, f32::NAN), (2.0, 3.0)];
        assert_eq!(value_near(&with_nan, 1.0, 2.0), None);
        assert_eq!(value_at(&with_nan, None, 0.0, 2.0), Some(3.0));
        let s = vec![Series::new("a", folo_ui::hsla(0., 0., 0., 1.), Arc::new(pts))];
        assert_eq!(cursor_x(&s, f32::NAN, 0.0, 2.0), None);
    }
}
