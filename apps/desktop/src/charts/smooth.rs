//! Data reduction: EMA smoothing and per-column envelopes, memoised per data revision.

use std::sync::Arc;

/// EMA smoothing with TensorBoard-style debiasing so the start isn't pulled toward zero.
pub fn ema(points: &[(f32, f32)], w: f32) -> Vec<(f32, f32)> {
    let mut last = 0.0;
    let mut out = Vec::with_capacity(points.len());
    for (i, &(x, y)) in points.iter().enumerate() {
        last = last * w + (1.0 - w) * y;
        let debias = 1.0 - w.powi(i as i32 + 1);
        out.push((x, last / debias));
    }
    out
}

/// Reduce to at most ~2 points per pixel column, keeping each column's min and max so spikes
/// survive downsampling.
fn envelope(points: &[(f32, f32)], x0: f32, x1: f32, columns: usize) -> Vec<(f32, f32)> {
    if points.len() <= columns * 2 || x1 <= x0 {
        return points.to_vec();
    }
    let mut out = Vec::with_capacity(columns * 2);
    let span = x1 - x0;
    let mut i = 0;
    for c in 0..columns {
        let end = x0 + span * (c + 1) as f32 / columns as f32;
        let start = i;
        let (mut lo, mut hi) = ((0.0, f32::INFINITY), (0.0, f32::NEG_INFINITY));
        while i < points.len() && (points[i].0 <= end || c == columns - 1) {
            let p = points[i];
            if p.1 < lo.1 {
                lo = p;
            }
            if p.1 > hi.1 {
                hi = p;
            }
            i += 1;
        }
        if i > start {
            if lo.0 <= hi.0 {
                out.push(lo);
                if hi != lo {
                    out.push(hi);
                }
            } else {
                out.push(hi);
                out.push(lo);
            }
        }
    }
    out
}

type Memo = Vec<(std::sync::Weak<Vec<(f32, f32)>>, [u32; 4], Arc<Vec<(f32, f32)>>)>;

thread_local! {
    /// Derived series keyed by their source allocation (held weakly, so a freed and reused
    /// allocation can never alias) plus the derivation parameters.
    static MEMO: std::cell::RefCell<Memo> = const { std::cell::RefCell::new(Vec::new()) };
}

pub(super) fn memo(
    src: &Arc<Vec<(f32, f32)>>,
    params: [u32; 4],
    f: impl FnOnce(&[(f32, f32)]) -> Vec<(f32, f32)>,
) -> Arc<Vec<(f32, f32)>> {
    MEMO.with(|m| {
        let mut m = m.borrow_mut();
        m.retain(|(w, _, _)| w.strong_count() > 0);
        if let Some((_, _, v)) =
            m.iter().find(|(w, p, _)| *p == params && w.upgrade().is_some_and(|a| Arc::ptr_eq(&a, src)))
        {
            return v.clone();
        }
        let v = Arc::new(f(src));
        if m.len() > 256 {
            m.remove(0);
        }
        m.push((Arc::downgrade(src), params, v.clone()));
        v
    })
}

/// The EMA of a series, computed once per data revision and weight.
pub fn smoothed(points: &Arc<Vec<(f32, f32)>>, w: f32) -> Arc<Vec<(f32, f32)>> {
    memo(points, [0, w.to_bits(), 0, 0], |p| ema(p, w))
}

/// The per-column envelope of a series for a plot width and x range, cached likewise.
pub(super) fn cached_envelope(
    points: &Arc<Vec<(f32, f32)>>,
    x0: f32,
    x1: f32,
    columns: usize,
) -> Arc<Vec<(f32, f32)>> {
    memo(points, [1, x0.to_bits(), x1.to_bits(), columns as u32], |p| envelope(p, x0, x1, columns))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ema_is_debiased() {
        let pts: Vec<(f32, f32)> = (0..10).map(|i| (i as f32, 5.0)).collect();
        assert!(ema(&pts, 0.9).iter().all(|p| (p.1 - 5.0).abs() < 1e-4));
    }

    #[test]
    fn envelope_keeps_extremes() {
        let mut pts: Vec<(f32, f32)> = (0..1000).map(|i| (i as f32, 1.0)).collect();
        pts[500].1 = 9.0;
        let e = envelope(&pts, 0.0, 999.0, 50);
        assert!(e.len() <= 100);
        assert!(e.iter().any(|p| p.1 == 9.0));
    }

    #[test]
    fn memo_follows_the_allocation() {
        let a = Arc::new(vec![(0.0, 1.0), (1.0, 3.0)]);
        let s1 = smoothed(&a, 0.5);
        assert!(Arc::ptr_eq(&s1, &smoothed(&a, 0.5)));
        let b = Arc::new(vec![(0.0, 1.0), (1.0, 3.0)]);
        assert!(!Arc::ptr_eq(&s1, &smoothed(&b, 0.5)));
        assert!(!Arc::ptr_eq(&s1, &smoothed(&a, 0.9)));
    }
}
