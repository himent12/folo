//! Axes: tick steps and positions, log axes, powers of ten and tick labels.

/// Round tick spacing near `range / n`: 1, 2, 2.5 or 5 × 10^k.
pub fn nice_step(range: f32, n: usize) -> f32 {
    if range <= 0.0 || !range.is_finite() {
        return 1.0;
    }
    let raw = range / n.max(1) as f32;
    let mag = 10f32.powf(raw.log10().floor());
    let r = raw / mag;
    let m = if r <= 1.0 {
        1.0
    } else if r <= 2.0 {
        2.0
    } else if r <= 2.5 {
        2.5
    } else if r <= 5.0 {
        5.0
    } else {
        10.0
    };
    m * mag
}

pub fn ticks(lo: f32, hi: f32, n: usize) -> Vec<f32> {
    let step = nice_step(hi - lo, n);
    let mut v = (lo / step).ceil() * step;
    let mut out = Vec::new();
    while v <= hi + step * 1e-3 && out.len() < 20 {
        out.push(if v.abs() < step * 1e-6 { 0.0 } else { v });
        v += step;
    }
    out
}

/// Format `v` with the decimals a tick step of `step` needs (0.25 → 2, 0.5 → 1, 10 → 0).
/// X-axis ticks that end exactly on the range when a round step divides it (0–120 → every 20),
/// otherwise the usual nice ticks.
pub fn even_ticks(lo: f32, hi: f32, max_n: usize) -> Vec<f32> {
    let range = hi - lo;
    if range > 0.0 && range.is_finite() {
        let mag = 10f32.powf((range / max_n as f32).log10().floor());
        for k in [1.0, 10.0, 100.0] {
            for m in [1.0, 2.0, 2.5, 5.0] {
                let step = m * mag * k;
                let n = range / step;
                if n <= max_n as f32
                    && n >= 2.0
                    && (n - n.round()).abs() < 1e-3
                    && (lo / step - (lo / step).round()).abs() < 1e-3
                {
                    return (0..=n.round() as usize).map(|i| lo + step * i as f32).collect();
                }
            }
        }
    }
    ticks(lo, hi, max_n)
}

/// Whole-number ticks for count axes: a 1-2-5 step that fits `max_n` labels, always ending on `hi`
/// (0–3 → 0 1 2 3; 0–120 → every 20; 0–7 → 0 5 7).
pub fn int_ticks(lo: f32, hi: f32, max_n: usize) -> Vec<f32> {
    let (lo, hi) = (lo.round(), hi.round().max(lo.round() + 1.0));
    let even = even_ticks(lo, hi, max_n);
    if even.iter().all(|v| v.fract() == 0.0) && even.last() == Some(&hi) {
        return even;
    }
    let range = hi - lo;
    // A step that divides the range exactly keeps every gap equal (0–120 → 0 40 80 120).
    for mag in [1.0f32, 10.0, 100.0, 1000.0, 10000.0] {
        for m in [1.0f32, 2.0, 3.0, 4.0, 5.0] {
            let step = m * mag;
            let n = range / step;
            if n.fract() == 0.0 && n >= 2.0 && n <= max_n.max(2) as f32 && (lo / step).fract() == 0.0 {
                return (0..=n as usize).map(|i| lo + step * i as f32).collect();
            }
        }
    }
    let mut step = 1.0f32;
    'find: for mag in [1.0f32, 10.0, 100.0, 1000.0, 10000.0] {
        for m in [1.0f32, 2.0, 5.0] {
            step = m * mag;
            if range / step <= max_n.max(1) as f32 {
                break 'find;
            }
        }
    }
    let mut out: Vec<f32> = Vec::new();
    let mut v = (lo / step).ceil() * step;
    while v < hi - 1e-3 {
        out.push(v);
        v += step;
    }
    // End on the range; drop the previous tick when the end would crowd it.
    if let Some(&last) = out.last()
        && hi - last < step * 0.6
        && out.len() > 1
    {
        out.pop();
    }
    out.push(hi);
    out
}

pub fn format_step(v: f32, step: f32) -> String {
    if step.abs() >= 1000.0 && v.abs() >= 1000.0 {
        return format_tick(v);
    }
    let mut d = 0usize;
    let mut s = step.abs();
    while d < 6 && (s - s.round()).abs() > 1e-3 * s.max(1e-9) {
        s *= 10.0;
        d += 1;
    }
    format!("{v:.d$}")
}

/// The stride (in octaves) of a log axis over `y0..y1` with room for `max` labels.
pub(super) fn pow2_stride(y0: f32, y1: f32, max: usize) -> i32 {
    let e0 = y0.max(1e-6).log2().ceil() as i32;
    let e1 = (y1.max(1e-6) * 1.0001).log2().floor() as i32;
    let n = (e1 - e0 + 1).max(1) as usize;
    n.div_ceil(max.max(1)) as i32
}

/// Powers of two from `y0` to `y1` (both > 0), for log axes, at most about `max` of them.
#[cfg(test)]
pub fn pow2_ticks(y0: f32, y1: f32, max: usize) -> Vec<f32> {
    pow2_ticks_every(y0, y1, pow2_stride(y0, y1, max))
}

/// [`pow2_ticks`] at a given stride of `k` octaves.
pub fn pow2_ticks_every(y0: f32, y1: f32, k: i32) -> Vec<f32> {
    let e0 = y0.max(1e-6).log2().ceil() as i32;
    let e1 = (y1.max(1e-6) * 1.0001).log2().floor() as i32;
    if e1 < e0 {
        return Vec::new();
    }
    // More powers than fit: every k-th, counted down from the top (8 2 0.5), so the top power
    // keeps its label and every step is the same factor.
    let k = k.max(1);
    (e0..=e1).filter(|e| (e1 - e) % k == 0).map(|e| 2f32.powi(e)).collect()
}

/// The power of ten a y axis over `y0..y1` with `ticks` ticks shows once, if its values are tiny
/// (tick step under 0.01). Legends of that chart format their values in the same power.
pub fn axis_pow(y0: f32, y1: f32, ticks: usize) -> Option<i32> {
    let step = nice_step(y1 - y0, ticks);
    // The top tick's power, not the step's: a 1.5×10⁻³ top reads "0 0.5 1.0 1.5 × 10⁻³", in the
    // power its values are written in elsewhere (New run's "1.11×10⁻³"), not "0 5 10 15 × 10⁻⁴".
    let top = y0.abs().max(y1.abs());
    (step > 0. && step < 0.01 && top > 0.).then(|| top.log10().floor() as i32)
}

/// A value in a given power of ten, as legends write it: two decimals ("6.71×10⁻⁴" reads
/// "0.67×10⁻³"), or two significant figures once it's below 0.1 ("0.011×10⁻³"), so a small
/// value never rounds to "0.00".
pub fn in_pow(x: f64, p: i32) -> String {
    let v = x * 10f64.powi(-p);
    let a = v.abs();
    let digits = if a == 0. || a >= 0.1 { 2 } else { (1 - a.log10().floor() as i32).max(2) as usize };
    format!("{v:.digits$}×10{}", superscript(p))
}

/// An exponent in superscript digits ("-3" → "⁻³").
pub fn superscript(p: i32) -> String {
    p.to_string()
        .chars()
        .map(|c| match c {
            '-' => '⁻',
            '0' => '⁰',
            '1' => '¹',
            '2' => '²',
            '3' => '³',
            '4' => '⁴',
            '5' => '⁵',
            '6' => '⁶',
            '7' => '⁷',
            '8' => '⁸',
            _ => '⁹',
        })
        .collect()
}

pub fn format_tick(v: f32) -> String {
    let a = v.abs();
    if a == 0.0 {
        "0".into()
    } else if a >= 10_000.0 {
        format!("{:.0}k", v / 1000.0)
    } else if a >= 1000.0 {
        format!("{:.1}k", v / 1000.0)
    } else if a >= 10.0 {
        format!("{v:.0}")
    } else if a >= 1.0 {
        format!("{v:.1}")
    } else if a >= 0.01 {
        format!("{v:.2}")
    } else {
        format!("{v:.0e}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nice_steps() {
        assert_eq!(nice_step(1.0, 4), 0.25);
        assert_eq!(nice_step(100.0, 5), 20.0);
        assert_eq!(nice_step(7.0, 4), 2.0);
        let t = ticks(0.0, 1.0, 4);
        assert_eq!(t, vec![0.0, 0.25, 0.5, 0.75, 1.0]);
        assert_eq!(format_step(0.5, 0.25), "0.50");
        assert_eq!(format_step(2.0, 1.0), "2");
        assert_eq!(format_step(0.004, 0.002), "0.004");
        assert_eq!(format_step(1.5, 0.5), "1.5");
        assert_eq!(even_ticks(0.0, 120.0, 8), vec![0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0]);
        assert_eq!(int_ticks(0.0, 3.0, 3), vec![0.0, 1.0, 2.0, 3.0]);
        assert_eq!(int_ticks(0.0, 120.0, 8), vec![0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0]);
        assert_eq!(int_ticks(0.0, 7.0, 3), vec![0.0, 7.0]);
        assert_eq!(int_ticks(0.0, 120.0, 3), vec![0.0, 40.0, 80.0, 120.0]);
        assert_eq!(int_ticks(0.0, 100.0, 3), vec![0.0, 50.0, 100.0]);
        assert_eq!(int_ticks(0.0, 103.0, 6), vec![0.0, 20.0, 40.0, 60.0, 80.0, 103.0]);
        assert!(int_ticks(0.0, 5.0, 2).iter().all(|v| v.fract() == 0.0));
        assert_eq!(even_ticks(0.0, 100.0, 5), vec![0.0, 20.0, 40.0, 60.0, 80.0, 100.0]);
        assert_eq!(even_ticks(0.0, 5.0, 8), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(even_ticks(0.0, 1.0, 5), vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0]);
    }

    #[test]
    fn log_axes_tick_at_powers_of_two() {
        assert_eq!(pow2_ticks(0.4, 9., 8), vec![0.5, 1., 2., 4., 8.]);
        assert_eq!(pow2_ticks(1., 1.5, 8), vec![1.]);
        // 0.001..10 is 13 powers; in room for 4, every fourth, down from 8.
        assert_eq!(pow2_ticks(0.001, 10., 4), vec![2f32.powi(-9), 2f32.powi(-5), 0.5, 8.]);
        // 0.25..8 in room for 3: every other power down from the top, even steps, domain kept.
        assert_eq!(pow2_stride(0.25, 8., 3), 2);
        assert_eq!(pow2_ticks(0.25, 8., 3), vec![0.5, 2., 8.]);
        // 0.001..2 is 11 powers: every third, down from 2.
        assert_eq!(pow2_ticks(0.001, 2., 4), vec![2f32.powi(-8), 2f32.powi(-5), 0.25, 2.]);
    }

    #[test]
    fn small_values_share_a_power_of_ten() {
        assert_eq!(superscript(-3), "⁻³");
        assert_eq!(axis_pow(0., 0.012, 3), Some(-2));
        assert_eq!(axis_pow(0., 0.0015, 3), Some(-3));
        assert_eq!(axis_pow(0., 1., 4), None);
        assert_eq!(in_pow(6.71e-4, -3), "0.67×10⁻³");
        assert_eq!(in_pow(1.11e-3, -3), "1.11×10⁻³");
        assert_eq!(in_pow(1.1e-5, -3), "0.011×10⁻³");
        assert_eq!(in_pow(6.7e-6, -3), "0.0067×10⁻³");
    }
}
