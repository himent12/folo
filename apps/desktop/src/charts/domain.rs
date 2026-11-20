//! Fitted y axes for 0–1 metrics (mAP, precision, recall), as Studio fits them
//! (`web/src/lib/charts/domain.ts`): from 0 to the highest value plus a little headroom, with 3 to 5
//! round gridlines, never a fixed 0–1 with the top third empty. The step and its decimals come from
//! the domain alone, so a short plot keeps every second (or fifth) tick of the same step instead of
//! picking a new one; an axis whose top is under [`UNIT_ZOOM_BELOW`] says it is zoomed in.

/// A fitted axis: its domain, the ticks it labels and the decimals every label takes.
#[derive(Clone, Debug, PartialEq)]
pub struct Fit {
    pub domain: (f32, f32),
    /// Tick values inside the domain (the top need not be one: 0–0.65 reads 0 · 0.2 · 0.4 · 0.6).
    pub ticks: Vec<f32>,
    /// Decimals of the tick step, for every label of the axis.
    pub decimals: usize,
}

/// Gridlines closer than this (px) read as a texture rather than a scale.
const MIN_GAP: f64 = 32.;
/// The most and fewest ticks a y axis takes.
pub const MAX_TICKS: usize = 5;
const MIN_TICKS: usize = 3;
/// Nice tick steps for a 0–1 metric, smallest first.
const UNIT_STEPS: [f64; 12] = [0.001, 0.002, 0.0025, 0.005, 0.01, 0.02, 0.025, 0.05, 0.1, 0.2, 0.25, 0.5];
/// How many of a step's ticks a short plot skips to keep its gridlines `MIN_GAP` apart.
const THIN: [usize; 3] = [1, 2, 5];

/// Below this top a 0–1 metric's axis is zoomed in, and the chart says so ([`UNIT_ZOOM_NOTE`]): a
/// run stuck near 0 must still read as low.
pub const UNIT_ZOOM_BELOW: f32 = 0.25;

/// The qualifier of a zoomed 0–1 axis: the scale it was cut from.
pub const UNIT_ZOOM_NOTE: &str = "zoomed in · full scale 0–1";

/// Decimals a tick step needs: 0.25 two, 0.2 one, 5 none (a tiny step as many as it takes).
pub fn step_decimals(step: f64) -> usize {
    if step.is_nan() || step <= 0. || !step.is_finite() {
        return 0;
    }
    let first = (-step.log10().floor()).max(0.) as usize;
    for d in first..=first + 8 {
        let x = step * 10f64.powi(d as i32);
        if (x - x.round()).abs() < 1e-6 * x.abs().max(1.) {
            return d;
        }
    }
    first + 8
}

/// The top of a 0–1 metric's axis: the highest value plus 5 % headroom, rounded up to a fifth of its
/// power of ten (0.02 from 0.1), at most 1, so the data reaches at least 80 % of the plot. It moves
/// in those steps, never with every value. No data: 1.
pub fn unit_top(max: f32) -> f32 {
    unit_top_f64(max) as f32
}

/// [`unit_top`] before it is narrowed for the plot: its ticks and their decimals are worked out
/// from this (narrowed first, 0.00000001 fell just under its power of ten and read 0.000000010).
fn unit_top_f64(max: f32) -> f64 {
    let max = max as f64;
    if !max.is_finite() || max <= 0. {
        return 1.;
    }
    let h = max * 1.05;
    let q = 10f64.powf(h.log10().floor()) / 5.;
    // Six significant digits, as Studio's `toPrecision(6)`: six decimal places made a top under
    // 0.0000005 zero (and cut a smaller one's digits).
    let top = ((h / q - 1e-9).ceil() * q).min(1.);
    let p = 10f64.powi(5 - top.log10().floor() as i32);
    (top * p).round() / p
}

/// A 0–1 metric's axis on a plot `plot_px` tall: from 0 to [`unit_top`]`(max)`, ticked at the nice
/// step whose highest tick comes nearest that top in 3–5 ticks (the smaller step on a tie). A plot
/// too short for those ticks 32 px apart keeps every 2nd (else 5th) one, same step, same decimals.
pub fn unit_domain(max: f32, plot_px: f32) -> Fit {
    let top = unit_top_f64(max);
    let mut best: Option<(f64, usize)> = None;
    for step in UNIT_STEPS {
        let n = (top / step + 1e-9).floor() as usize + 1;
        if !(MIN_TICKS..=MAX_TICKS).contains(&n) {
            continue;
        }
        if best.is_none_or(|(s, k)| (n - 1) as f64 * step > (k - 1) as f64 * s + 1e-9) {
            best = Some((step, n));
        }
    }
    let Some((step, n)) = best else {
        // A top too small for three nice ticks: its base and its top.
        return Fit { domain: (0., top as f32), ticks: vec![0., top as f32], decimals: step_decimals(top) };
    };
    let d = step_decimals(step);
    let round = |v: f64| ((v * 10f64.powi(d as i32)).round() / 10f64.powi(d as i32)) as f32;
    let all: Vec<f32> = (0..n).map(|k| round(k as f64 * step)).collect();
    let every = THIN.into_iter().find(|&k| k < n && plot_px as f64 * step * k as f64 / top >= MIN_GAP);
    let ticks = match every {
        Some(k) => all.iter().enumerate().filter(|(i, _)| i % k == 0).map(|(_, v)| *v).collect(),
        // Not even every fifth tick fits: the base and the highest tick.
        None => vec![0., all[n - 1]],
    };
    Fit { domain: (0., top as f32), ticks, decimals: d }
}

/// Whether a 0–1 axis fitted to `max` is zoomed in (its top under [`UNIT_ZOOM_BELOW`]).
pub fn unit_zoomed(max: f32) -> bool {
    unit_top(max) < UNIT_ZOOM_BELOW - 1e-6
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    #[test]
    fn fits_a_unit_metric_to_its_data_plus_a_little_headroom() {
        // A failed run's mAP50-95, a comparison's top, a completed run's mAP50, a run that learned little.
        assert!(close(unit_domain(0.494, 290.).domain.1, 0.52));
        assert!(close(unit_domain(0.606, 360.).domain.1, 0.64));
        assert!(close(unit_domain(0.803, 122.).domain.1, 0.86));
        assert!(close(unit_domain(0.081, 170.).domain.1, 0.086));
        // The data reaches at least 80 % of the plot's height, whatever the value.
        for max in [0.004, 0.0099, 0.01, 0.037, 0.081, 0.1, 0.12, 0.27, 0.45, 0.5, 0.63, 0.71, 0.88, 0.97] {
            assert!(max / unit_domain(max, 290.).domain.1 > 0.8, "{max}");
        }
        assert_eq!(unit_top(0.99), 1.);
    }

    /// #382: the top was rounded to six decimal places, so a score under 0.0000005 got a top of 0
    /// and a small one lost digits. Six significant digits, as Studio's `unitTop`.
    #[test]
    fn a_tiny_metric_keeps_its_top() {
        let near = |a: f32, b: f32| ((a - b) / b).abs() < 1e-5;
        assert!(near(unit_top(0.000_000_37), 0.000_000_4), "{}", unit_top(0.000_000_37));
        assert!(near(unit_top(0.000_003_3), 0.000_003_6), "{}", unit_top(0.000_003_3));
        assert!(near(unit_top(0.000_068), 0.000_072), "{}", unit_top(0.000_068));
        assert!(close(unit_top(0.494), 0.52));
        // Review round 1: a top of 1e-8 is labelled as Studio labels it, not 0.000000010.
        assert_eq!(unit_domain(0.000_000_009_4, 290.).decimals, 8);
    }

    #[test]
    fn draws_three_to_five_nice_gridlines_the_highest_near_the_top() {
        assert_eq!(unit_domain(0.494, 290.).ticks, vec![0., 0.25, 0.5]);
        assert_eq!(unit_domain(0.606, 360.).ticks, vec![0., 0.2, 0.4, 0.6]);
        let f = unit_domain(0.803, 290.);
        assert_eq!(f.ticks, vec![0., 0.2, 0.4, 0.6, 0.8]);
        assert_eq!(f.decimals, 1);
        assert_eq!(unit_domain(0.081, 170.).ticks, vec![0., 0.02, 0.04, 0.06, 0.08]);
        for max in [0.004, 0.01, 0.12, 0.27, 0.45, 0.5, 0.63, 0.71, 0.88, 0.97, 1.] {
            let f = unit_domain(max, 360.);
            assert!((MIN_TICKS..=MAX_TICKS).contains(&f.ticks.len()), "{max}: {:?}", f.ticks);
            let top = *f.ticks.last().unwrap();
            assert!(top / f.domain.1 >= 0.7, "{max} top tick");
            assert!(top <= f.domain.1 + 1e-6, "{max}");
        }
    }

    #[test]
    fn keeps_one_step_at_every_plot_height_a_short_plot_shows_every_second_tick() {
        assert_eq!(unit_domain(0.803, 122.).ticks, vec![0., 0.4, 0.8]);
        for max in [0.004, 0.0099, 0.037, 0.081, 0.12, 0.27, 0.45, 0.5, 0.63, 0.71, 0.803, 0.88, 0.97, 1.] {
            let wide = unit_domain(max, 360.);
            let step = wide.ticks[1];
            for h in [60., 96., 110., 122., 138., 170., 210., 290.] {
                let f = unit_domain(max, h);
                assert_eq!(f.domain, wide.domain, "{max} @ {h}");
                assert_eq!(f.decimals, wide.decimals, "{max} @ {h}");
                // Every tick a short plot shows is one of the tall plot's.
                for t in &f.ticks {
                    assert!(wide.ticks.contains(t), "{max} @ {h}: {t}");
                }
                let gap = f.ticks[1] - f.ticks[0];
                assert!(((gap / step).round() * step - gap).abs() < 1e-5, "{max} @ {h}");
                if h * 2. * step / wide.domain.1 >= 32. {
                    assert!(h * gap / wide.domain.1 >= 32. - 1e-3, "{max} @ {h} gap");
                }
                assert!(f.ticks.len() >= 2);
            }
        }
    }

    #[test]
    fn says_when_it_is_zoomed_in() {
        assert!(unit_zoomed(0.081));
        assert!(unit_zoomed(0.2));
        assert!(!unit_zoomed(0.3));
    }

    #[test]
    fn never_goes_past_1_and_falls_back_to_0_1_without_data() {
        let f = unit_domain(1., 290.);
        assert_eq!(f, Fit { domain: (0., 1.), ticks: vec![0., 0.25, 0.5, 0.75, 1.], decimals: 2 });
        assert_eq!(unit_domain(f32::NAN, 290.).domain, (0., 1.));
        assert_eq!(unit_domain(0., 290.).domain, (0., 1.));
    }

    #[test]
    fn is_stable_a_lower_value_never_shrinks_it_a_higher_one_moves_it_in_steps() {
        let mut best = 0f32;
        let seen: Vec<f32> = [0.31, 0.42, 0.48, 0.47, 0.5, 0.55, 0.57]
            .iter()
            .map(|&v| {
                best = best.max(v);
                unit_domain(best, 290.).domain.1
            })
            .collect();
        let want = [0.34, 0.46, 0.52, 0.52, 0.54, 0.58, 0.6];
        assert!(seen.iter().zip(want).all(|(a, b)| close(*a, b)), "{seen:?}");
    }

    #[test]
    fn counts_the_decimals_a_step_needs() {
        assert_eq!(step_decimals(0.25), 2);
        assert_eq!(step_decimals(0.2), 1);
        assert_eq!(step_decimals(5.), 0);
        assert_eq!(step_decimals(0.0025), 4);
        assert_eq!(step_decimals(2e-7), 7);
    }
}
