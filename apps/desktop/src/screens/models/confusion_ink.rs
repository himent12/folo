//! The confusion matrix's one luminance ramp and its rankings, as Studio draws them
//! (`web/src/lib/charts/confusionInk.ts`, `web/src/routes/model/confusion.ts`).
//!
//! A cell is shaded by its share of the row: ink (`text.primary`) mixed into the panel in oklab, so
//! the steps are perceptually even. Small shares are where a matrix's story is, so the ramp's floor is
//! stepped: under 1 %, 1–5 % and 5–10 % each get a step of their own, and from 10 % it runs evenly to
//! the end. A cell with no objects is the panel itself, edged by the hairline grid. The diagonal is
//! told apart by its place, not a hue. Labels are pure white or black, whichever reads better (the
//! near-black text ink falls just short of 4.5:1 on the ramp's middle in dark mode).

use folo_ui::{Hsla, Rgba};

use crate::theme::{Mode, Theme};

/// Where the floor steps start, as shares of the row: under 1 %, then 1 %, 5 % and the even ramp from
/// 10 %.
pub const FLOOR: [f32; 3] = [0.01, 0.05, 0.1];

/// Ink share mixed into the panel: one per floor step, then from `from` at 10 % evenly to `end` at
/// 100 %. Light starts lower: on white a heavier first step turns a matrix of 1–4 % confusions into a
/// grey chequerboard, while on dark the step must clear a near-black panel.
struct Ramp {
    steps: [f32; 3],
    from: f32,
    end: f32,
}

const DARK: Ramp = Ramp { steps: [0.2, 0.3, 0.4], from: 0.48, end: 0.94 };
const LIGHT: Ramp = Ramp { steps: [0.12, 0.2, 0.28], from: 0.36, end: 0.94 };

/// Where a cell of share `v` sits on the ramp, as the ink share mixed into the panel; 0 when empty.
pub fn ramp_share(v: f32, mode: Mode) -> f32 {
    if v.is_nan() || v <= 0. {
        return 0.;
    }
    let r = if mode == Mode::Dark { &DARK } else { &LIGHT };
    if v < FLOOR[0] {
        r.steps[0]
    } else if v < FLOOR[1] {
        r.steps[1]
    } else if v < FLOOR[2] {
        r.steps[2]
    } else {
        r.from + (v.min(1.) - FLOOR[2]) / (1. - FLOOR[2]) * (r.end - r.from)
    }
}

/// A cell's fill at share `v`: ink mixed into the panel; the panel itself when empty.
pub fn cell_fill(t: &Theme, v: f32) -> Hsla {
    let s = ramp_share(v, t.mode);
    if s <= 0. { t.bg_panel } else { mix_oklab(t.text_primary, t.bg_panel, s) }
}

/// A cell label's ink on `fill`: pure white or pure black, whichever contrasts more.
pub fn label_ink(fill: Hsla) -> Hsla {
    let l = luminance(fill);
    if (1.05) / (l + 0.05) >= (l + 0.05) / 0.05 { folo_ui::white() } else { folo_ui::black() }
}

fn luminance(c: Hsla) -> f32 {
    let c: Rgba = c.into();
    0.2126 * lin(c.r) + 0.7152 * lin(c.g) + 0.0722 * lin(c.b)
}

/// A share of the row as the matrix writes it, one rule everywhere (cells, readout, lists): whole
/// percentages, "<1" for a share above zero that rounds below 1, "" for none. No "%" sign.
pub fn share_pct(v: f32) -> String {
    if v.is_nan() || v <= 0. {
        String::new()
    } else if v < 0.005 {
        "<1".into()
    } else {
        format!("{}", (v * 100.).round().max(1.) as u32)
    }
}

/// `a` mixed into `b` at share `s` of `a`, in oklab (CSS `color-mix(in oklab, a s, b)`).
pub fn mix_oklab(a: Hsla, b: Hsla, s: f32) -> Hsla {
    let (x, y) = (oklab(a.into()), oklab(b.into()));
    let m = [0, 1, 2].map(|i| x[i] * s + y[i] * (1. - s));
    let mut out = from_oklab(m);
    out.a = 1.;
    out.into()
}

fn lin(v: f32) -> f32 {
    if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) }
}

fn gamma(v: f32) -> f32 {
    let v = v.clamp(0., 1.);
    if v <= 0.003_130_8 { v * 12.92 } else { 1.055 * v.powf(1. / 2.4) - 0.055 }
}

fn oklab(c: Rgba) -> [f32; 3] {
    let (r, g, b) = (lin(c.r), lin(c.g), lin(c.b));
    let l = (0.412_221_46 * r + 0.536_332_55 * g + 0.051_445_995 * b).cbrt();
    let m = (0.211_903_5 * r + 0.680_699_5 * g + 0.107_396_96 * b).cbrt();
    let s = (0.088_302_46 * r + 0.281_718_85 * g + 0.629_978_7 * b).cbrt();
    [
        0.210_454_26 * l + 0.793_617_8 * m - 0.004_072_047 * s,
        1.977_998_5 * l - 2.428_592_2 * m + 0.450_593_7 * s,
        0.025_904_037 * l + 0.782_771_77 * m - 0.808_675_77 * s,
    ]
}

fn from_oklab([l, a, b]: [f32; 3]) -> Rgba {
    let l_ = (l + 0.396_337_78 * a + 0.215_803_76 * b).powi(3);
    let m_ = (l - 0.105_561_346 * a - 0.063_854_17 * b).powi(3);
    let s_ = (l - 0.089_484_18 * a - 1.291_485_5 * b).powi(3);
    Rgba {
        r: gamma(4.076_741_7 * l_ - 3.307_711_6 * m_ + 0.230_969_94 * s_),
        g: gamma(-1.268_438 * l_ + 2.609_757_4 * m_ - 0.341_319_38 * s_),
        b: gamma(-0.004_196_086_3 * l_ - 0.703_418_6 * m_ + 1.707_614_7 * s_),
        a: 1.,
    }
}

/// What a ranked line is: a true class taken for another, or missed (its objects matched nothing).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Confused,
    Missed,
}

/// One line of the Most confused / Most missed lists.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Entry {
    pub kind: Kind,
    /// True class (matrix row).
    pub r: usize,
    /// Predicted class (matrix column); the Missed column for a miss.
    pub c: usize,
    pub count: u32,
    /// The true class's objects (its row's sum).
    pub total: u32,
    /// count / total.
    pub v: f32,
}

/// How many lines each list shows.
pub const RANK_MAX: usize = 5;

/// The strongest confusions and misses of an (n+1)×(n+1) matrix (last row and column: no object /
/// missed), largest share of the true class first (ties: more objects first). Rows with fewer than
/// `min_row` objects are skipped, so one stray box can't top the list; the false-positive row is not
/// a true class and is left out.
pub fn rank_confusions(conf: &[Vec<u32>], limit: usize, min_row: u32) -> Vec<Entry> {
    let k = conf.len().saturating_sub(1);
    let mut out = Vec::new();
    for (r, row) in conf.iter().enumerate().take(k) {
        let total: u32 = row.iter().sum();
        if total < min_row {
            continue;
        }
        for (c, &count) in row.iter().enumerate().take(k + 1) {
            if c == r || count == 0 {
                continue;
            }
            let kind = if c == k { Kind::Missed } else { Kind::Confused };
            out.push(Entry { kind, r, c, count, total, v: count as f32 / total as f32 });
        }
    }
    out.sort_by(|a, b| b.v.total_cmp(&a.v).then(b.count.cmp(&a.count)));
    out.truncate(limit);
    out
}

/// The share of every true object the model missed (the Missed column over all class rows); None
/// without objects.
pub fn missed_overall(conf: &[Vec<u32>]) -> Option<f32> {
    let k = conf.len().checked_sub(1)?;
    let (mut total, mut miss) = (0u64, 0u64);
    for row in conf.iter().take(k) {
        total += row.iter().map(|&v| v as u64).sum::<u64>();
        miss += row.get(k).copied().unwrap_or(0) as u64;
    }
    (total > 0).then(|| miss as f32 / total as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contrast(a: Hsla, b: Hsla) -> f32 {
        let (x, y) = (luminance(a), luminance(b));
        (x.max(y) + 0.05) / (x.min(y) + 0.05)
    }

    #[test]
    fn the_ramp_steps_its_floor_then_runs_evenly_to_its_end() {
        assert_eq!(ramp_share(0., Mode::Dark), 0.);
        assert_eq!(ramp_share(0.004, Mode::Dark), 0.2);
        assert_eq!(ramp_share(0.02, Mode::Dark), 0.3);
        assert_eq!(ramp_share(0.07, Mode::Dark), 0.4);
        assert!((ramp_share(0.1, Mode::Dark) - 0.48).abs() < 1e-6);
        assert!((ramp_share(1., Mode::Dark) - 0.94).abs() < 1e-6);
        assert_eq!(ramp_share(0.004, Mode::Light), 0.12);
        assert!((ramp_share(1., Mode::Light) - 0.94).abs() < 1e-6);
        // A 2 % and an 8 % confusion never look alike; the ramp only rises.
        let mut last = 0.;
        for i in 1..=1000 {
            let s = ramp_share(i as f32 / 1000., Mode::Light);
            assert!(s >= last);
            last = s;
        }
    }

    #[test]
    fn every_cell_keeps_its_label_readable_and_cells_darken_with_their_share() {
        for t in [Theme::new(Mode::Dark), Theme::new(Mode::Light)] {
            let mut prev = contrast(cell_fill(&t, 0.), t.bg_panel);
            for i in 0..=1000 {
                let fill = cell_fill(&t, i as f32 / 1000.);
                let c = contrast(label_ink(fill), fill);
                assert!(c >= 4.5, "{:?} cell at {}: {c:.2}:1", t.mode, i as f32 / 1000.);
                // Further from the panel as the share grows.
                let d = contrast(fill, t.bg_panel);
                assert!(d >= prev - 1e-3, "{:?} at {i}", t.mode);
                prev = d;
            }
            assert_eq!(cell_fill(&t, 0.), t.bg_panel);
        }
    }

    #[test]
    fn oklab_round_trips_and_mixes_between_its_ends() {
        let t = Theme::new(Mode::Light);
        let back: Rgba = from_oklab(oklab(t.text_primary.into()));
        let want: Rgba = t.text_primary.into();
        assert!(
            (back.r - want.r).abs() < 2e-3
                && (back.g - want.g).abs() < 2e-3
                && (back.b - want.b).abs() < 2e-3
        );
        let (end, panel): (Rgba, Rgba) =
            (mix_oklab(t.text_primary, t.bg_panel, 0.).into(), t.bg_panel.into());
        assert!((end.r - panel.r).abs() < 1e-3 && (end.b - panel.b).abs() < 1e-3);
    }

    #[test]
    fn shares_read_in_whole_percent_with_lt1_for_the_smallest() {
        assert_eq!(share_pct(0.), "");
        assert_eq!(share_pct(0.004), "<1");
        assert_eq!(share_pct(0.005), "1");
        assert_eq!(share_pct(0.126), "13");
        assert_eq!(share_pct(1.), "100");
    }

    #[test]
    fn ranks_by_share_of_the_true_class_skipping_small_rows_and_false_positives() {
        // Three classes and the no-object row/column.
        let conf = vec![
            vec![80, 10, 0, 10], // 100 objects: 10 % taken for 1, 10 % missed
            vec![5, 30, 0, 15],  // 50: 10 % taken for 0, 30 % missed
            vec![0, 5, 3, 2],    // 10 objects: under 20, skipped
            vec![9, 9, 9, 0],    // false positives: never a true class
        ];
        let all = rank_confusions(&conf, usize::MAX, 20);
        let got: Vec<(Kind, usize, usize)> = all.iter().map(|e| (e.kind, e.r, e.c)).collect();
        assert_eq!(
            got,
            vec![(Kind::Missed, 1, 3), (Kind::Confused, 0, 1), (Kind::Missed, 0, 3), (Kind::Confused, 1, 0)]
        );
        // Equal shares: more objects first.
        assert_eq!((all[1].count, all[1].total), (10, 100));
        assert_eq!(rank_confusions(&conf, 2, 20).len(), 2);
        assert_eq!(rank_confusions(&conf, usize::MAX, 1).len(), 6);
        assert!(rank_confusions(&[vec![0]], 5, 20).is_empty());
        assert!((missed_overall(&conf).unwrap() - 27. / 160.).abs() < 1e-6);
    }
}
