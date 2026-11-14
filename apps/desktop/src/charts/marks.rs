//! The marks at a line's end and on its best point, as Studio draws them
//! (`web/src/lib/charts/endMarks.ts`): each mark has one shape for one meaning. The best epoch is a
//! hollow ring in ink; while a run is live, the end of a chart's first series a filled lime dot and
//! every other line's a dot of its own hue; a completed run's lines end in a dot of their hue, a
//! stopped run's in a square and a failed run's in a cross. So lime on a chart always means now,
//! once per chart. The glyph says the status and the hue says whose line it is: a cross on a hued
//! line takes the line's hue, one on an ink line the danger ink. The best ring and the end mark are
//! always two shapes; only when the best is the line's end (or so close that the marks would touch)
//! is the ring drawn around the end mark, as one mark.

use folo_ui::{Bounds, Hsla, PathBuilder, Pixels, Point, Window, fill, point, px, size};

use crate::theme::Theme;

/// How a line ends.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum End {
    /// No mark (a curve that is not a run's history).
    #[default]
    None,
    /// A run still training: a filled lime dot (a chart's first series only; see `End::trailing`).
    Live,
    /// A completed run: a dot of the line's hue.
    Done,
    /// A stopped run: a square.
    Stopped,
    /// A failed run: a cross.
    Failed,
}

impl End {
    /// The end of a chart's other series: a live run's line that is not the chart's first ends in
    /// a dot of its own hue, so the chart has one lime dot. Any other end is unchanged.
    pub fn trailing(self) -> Self {
        if self == End::Live { End::Done } else { self }
    }
}

/// The ends as drawn: the first live end keeps its lime dot, every later one ends in its hue.
pub fn one_live(ends: impl IntoIterator<Item = End>) -> Vec<End> {
    let mut lime = false;
    ends.into_iter()
        .map(|e| {
            if e != End::Live {
                e
            } else if lime {
                e.trailing()
            } else {
                lime = true;
                e
            }
        })
        .collect()
}

/// The end dot's radius, px.
pub const END_R: f32 = 3.;
/// How much larger a live run's lime dot is than the end dot.
pub const LIVE_GROW: f32 = 1.;
/// The best ring's radius, px (a 4.5 px ring, 1.5 px wide).
pub const BEST_R: f32 = 5.25;
/// A best ring with less clear space than this (px, edge to edge) to an end mark is drawn around it.
pub const MERGE_PX: f32 = 2.;

/// An end mark's outer radius.
pub fn end_radius(end: End) -> f32 {
    if end == End::Live { END_R + LIVE_GROW } else { END_R }
}

/// The index of the line end (x, y, outer radius) the best ring at `best` is drawn around: the
/// nearest with less than `MERGE_PX` of clear space between the two marks, else None.
pub fn merged_end(best: (f32, f32), ends: &[(f32, f32, f32)]) -> Option<usize> {
    let mut found = None;
    let mut d = MERGE_PX;
    for (i, &(x, y, r)) in ends.iter().enumerate() {
        let k = (x - best.0).hypot(y - best.1) - BEST_R - r;
        if k < d {
            d = k;
            found = Some(i);
        }
    }
    found
}

/// The ink of an end mark on a line drawn in `line`: lime for a live end, the danger ink for a failed
/// end on an ink line (`ink`), the line's own colour otherwise.
pub fn end_ink(t: &Theme, end: End, line: Hsla, ink: bool) -> Hsla {
    match end {
        End::Live => t.accent_fill(),
        End::Failed if ink => t.danger_text,
        _ => line,
    }
}

/// Whether `c` is one of the text inks (a line in ink rather than a series hue).
pub fn is_ink(t: &Theme, c: Hsla) -> bool {
    c == t.text_primary || c == t.text_secondary || c == t.text_tertiary
}

fn disc(window: &mut Window, c: Point<Pixels>, r: f32, color: Hsla) {
    window.paint_quad(fill(Bounds::centered_at(c, size(px(2. * r), px(2. * r))), color).corner_radii(px(r)));
}

/// Paints a line's end mark at `c`, over a halo in the panel colour.
pub fn paint_end(window: &mut Window, t: &Theme, c: Point<Pixels>, end: End, color: Hsla) {
    let r = end_radius(end);
    match end {
        End::None => {}
        End::Live | End::Done => {
            disc(window, c, r + 1.5, t.bg_panel);
            disc(window, c, r, color);
        }
        End::Stopped => {
            let halo = Bounds::centered_at(c, size(px(2. * r + 3.), px(2. * r + 3.)));
            window.paint_quad(fill(halo, t.bg_panel));
            window.paint_quad(fill(Bounds::centered_at(c, size(px(2. * r), px(2. * r))), color));
        }
        End::Failed => {
            for (w, ink) in [(4.5, t.bg_panel), (2., color)] {
                for (a, b) in [((-r, -r), (r, r)), ((r, -r), (-r, r))] {
                    let mut p = PathBuilder::stroke(px(w));
                    p.move_to(point(c.x + px(a.0), c.y + px(a.1)));
                    p.line_to(point(c.x + px(b.0), c.y + px(b.1)));
                    if let Ok(path) = p.build() {
                        window.paint_path(path, ink);
                    }
                }
            }
        }
    }
}

/// Paints the best ring: on its own (filled with the panel, so the line doesn't run through it), or
/// around the end mark of radius `around` it merged with.
pub fn paint_best(window: &mut Window, t: &Theme, c: Point<Pixels>, around: Option<f32>) {
    let r = around.map_or(4.5, |r| r + 3.);
    let b = Bounds::centered_at(c, size(px(2. * r), px(2. * r)));
    let bg = if around.is_some() { folo_ui::transparent_black() } else { t.bg_panel };
    window.paint_quad(folo_ui::quad(b, px(r), bg, px(1.5), t.text_primary, folo_ui::BorderStyle::Solid));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::theme::Mode;

    #[test]
    fn a_best_ring_goes_around_an_end_mark_only_where_the_two_would_touch() {
        let ends = [(100., 50., 4.), (300., 50., 3.)];
        // Best at epoch 36, live at 37, 18 px apart: a ring and a dot, never "oo".
        assert_eq!(merged_end((82., 50.), &ends), None);
        // The best is the latest: the ring goes around the dot.
        assert_eq!(merged_end((100., 50.), &ends), Some(0));
        // Epochs 36 and 37 three px apart on a narrow plot would overlap: one mark.
        assert_eq!(merged_end((97., 50.), &ends), Some(0));
        let edge = BEST_R + 4. + MERGE_PX;
        assert_eq!(merged_end((100. - edge + 0.1, 50.), &ends), Some(0));
        assert_eq!(merged_end((100. - edge, 50.), &ends), None);
        assert_eq!(merged_end((299., 50.), &ends), Some(1));
        assert_eq!(merged_end((10., 10.), &ends), None);
    }

    #[test]
    fn a_live_end_is_lime_a_failure_on_an_ink_line_danger_any_other_end_its_line() {
        for t in [Theme::new(Mode::Dark), Theme::new(Mode::Light)] {
            let hue = t.chart_series[3];
            assert_eq!(end_ink(&t, End::Live, hue, false), t.accent_fill());
            assert_eq!(end_ink(&t, End::Live, t.text_primary, true), t.accent_fill());
            assert_eq!(end_ink(&t, End::Failed, t.text_primary, true), t.danger_text);
            // On a hued line the glyph says it failed; the hue still says whose line it is.
            assert_eq!(end_ink(&t, End::Failed, hue, false), hue);
            assert_eq!(end_ink(&t, End::Stopped, hue, false), hue);
            assert_eq!(end_ink(&t, End::Done, hue, false), hue);
            assert!(is_ink(&t, t.text_primary) && is_ink(&t, t.text_tertiary) && !is_ink(&t, hue));
        }
        assert_eq!(end_radius(End::Live), END_R + LIVE_GROW);
        assert_eq!(end_radius(End::Stopped), END_R);
    }

    #[test]
    fn a_chart_has_one_lime_end_and_its_other_live_lines_end_in_their_hue() {
        use End::*;
        assert_eq!(Live.trailing(), Done);
        for e in [None, Done, Stopped, Failed] {
            assert_eq!(e.trailing(), e);
        }
        // Whatever the call sites ask for, a chart draws at most one lime dot: the first.
        assert_eq!(one_live([Live, Live, Live]), vec![Live, Done, Done]);
        assert_eq!(one_live([None, Done, Live, Live]), vec![None, Done, Live, Done]);
        // A finished run keeps every outcome mark.
        assert_eq!(one_live([Stopped, Stopped]), vec![Stopped, Stopped]);
        assert_eq!(one_live([Failed, Failed]), vec![Failed, Failed]);
    }
}
