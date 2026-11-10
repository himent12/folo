//! Easing live data into place between revisions.

use std::sync::Arc;

use gpui::{SharedString, Window};

use super::spec::LineChart;

/// How long live points take to move into place (DIRECTION: motion).
const EASE_MS: f32 = 240.;

/// Live series longer than this jump instead of easing: a new point among thousands moves no
/// pixel, and re-deriving a long history every frame isn't worth it.
const EASE_MAX_POINTS: usize = 512;

/// A series's points, shared between renders.
pub type Points = Arc<Vec<(f32, f32)>>;
type Pts = Points;

/// A chart's last data and the move from the previous data to it.
struct Ease {
    /// What the data belongs to (run or model, and the series names): a change means new data.
    key: (u64, Vec<SharedString>),
    /// Per series: the points it moves from, or None when it isn't moving.
    from: Vec<Option<Pts>>,
    to: Vec<Pts>,
    start: std::time::Instant,
}

thread_local! {
    static EASES: std::cell::RefCell<std::collections::HashMap<SharedString, Ease>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
}

/// Points between two revisions of a series at progress `e`: shared points move vertically, new
/// points slide out from the old last point.
pub fn tween(from: &[(f32, f32)], to: &[(f32, f32)], e: f32) -> Vec<(f32, f32)> {
    let lerp = |a: f32, b: f32| a + (b - a) * e;
    to.iter()
        .enumerate()
        .map(|(i, &(x, y))| match from.get(i) {
            Some(&(_, fy)) => (x, lerp(fy, y)),
            None => from.last().map_or((x, y), |&(lx, ly)| (lerp(lx, x), lerp(ly, y))),
        })
        .collect()
}

/// An update that only appends or revises the tail of the same series (a live run), as opposed
/// to different data (another run, a toggled series), which should jump.
fn is_append(from: &[(f32, f32)], to: &[(f32, f32)]) -> bool {
    !from.is_empty()
        && to.len() >= from.len()
        && to.len() - from.len() <= 8.max(from.len() / 20)
        && to[0].0 == from[0].0
        && to[from.len() - 1].0 == from[from.len() - 1].0
}

/// Same points, bit for bit (a rebuilt allocation of unchanged data counts as the same).
fn same_points(a: &Pts, b: &Pts) -> bool {
    Arc::ptr_eq(a, b)
        || (a.len() == b.len()
            && a.iter()
                .zip(b.iter())
                .all(|(p, q)| p.0.to_bits() == q.0.to_bits() && p.1.to_bits() == q.1.to_bits()))
}

/// A stable key for what a chart's data belongs to (a run or model id).
pub fn ease_key(id: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    id.hash(&mut h);
    h.finish()
}

/// One frame of easing for chart `id`: the points to draw instead of `to` while a move runs.
fn ease_frame(
    id: &SharedString,
    key: (u64, Vec<SharedString>),
    to: &[Pts],
    now: std::time::Instant,
    reduce: bool,
) -> Option<Vec<Pts>> {
    let progress = |start: std::time::Instant| (now - start).as_secs_f32() * 1000. / EASE_MS;
    let settled = now - std::time::Duration::from_secs(1);
    EASES.with(|m| {
        let mut m = m.borrow_mut();
        let e = m.entry(id.clone()).or_insert_with(|| Ease {
            key: key.clone(),
            from: vec![None; to.len()],
            to: to.to_vec(),
            start: settled,
        });
        if reduce {
            // Reduce motion: no move starts, and one in progress stops where its data is.
            *e = Ease { key, from: vec![None; to.len()], to: to.to_vec(), start: settled };
            return None;
        }
        let unchanged =
            e.key == key && e.to.len() == to.len() && e.to.iter().zip(to).all(|(a, b)| same_points(a, b));
        if unchanged {
            // Same data, perhaps rebuilt: adopt the new allocations; a move in progress goes on.
            e.to = to.to_vec();
        } else {
            let p = progress(e.start).min(1.);
            let related = e.key == key && e.to.len() == to.len() && e.from.len() == to.len();
            // One slot per series, always: a hidden-then-restored series can ease again.
            let from: Vec<Option<Pts>> = if !related {
                vec![None; to.len()]
            } else {
                e.from
                    .iter()
                    .zip(&e.to)
                    .zip(to)
                    .map(|((f, old), new)| {
                        if same_points(old, new) {
                            return None;
                        }
                        // Start from what is on screen now (mid-move if this series was moving).
                        let cur = match f {
                            Some(f) if p < 1. => tween(f, old, ease_out(p)),
                            _ => old.to_vec(),
                        };
                        (new.len() <= EASE_MAX_POINTS && is_append(&cur, new)).then(|| Arc::new(cur))
                    })
                    .collect()
            };
            let moving = from.iter().any(Option::is_some);
            *e = Ease { key, from, to: to.to_vec(), start: if moving { now } else { settled } };
        }
        let p = progress(e.start);
        if p >= 1. || e.from.iter().all(Option::is_none) {
            return None;
        }
        let k = ease_out(p);
        // Series that aren't moving keep their allocation, so their cached derivations hold.
        Some(
            e.from
                .iter()
                .zip(&e.to)
                .map(|(f, t)| f.as_ref().map_or(t.clone(), |f| Arc::new(tween(f, t, k))))
                .collect(),
        )
    })
}

/// Ease new live data in over 240 ms (instant under Reduce motion). Numbers elsewhere are not
/// animated; only this chart's geometry is.
pub(super) fn ease(spec: &mut LineChart, window: &mut Window) {
    let key = (spec.ease_key, spec.series.iter().map(|s| s.name.clone()).collect());
    let to: Vec<Pts> = spec.series.iter().map(|s| s.points.clone()).collect();
    if let Some(pts) = ease_frame(&spec.id, key, &to, std::time::Instant::now(), crate::ui::reduce_motion()) {
        for (s, p) in spec.series.iter_mut().zip(pts) {
            s.points = p;
        }
        window.request_animation_frame();
    }
}

fn ease_out(p: f32) -> f32 {
    1. - (1. - p.clamp(0., 1.)).powi(3)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tween_moves_shared_points_and_slides_new_ones() {
        let from = [(0.0, 0.0), (1.0, 1.0)];
        let to = [(0.0, 0.0), (1.0, 3.0), (2.0, 5.0)];
        assert_eq!(tween(&from, &to, 0.0), vec![(0.0, 0.0), (1.0, 1.0), (1.0, 1.0)]);
        assert_eq!(tween(&from, &to, 0.5), vec![(0.0, 0.0), (1.0, 2.0), (1.5, 3.0)]);
        assert_eq!(tween(&from, &to, 1.0), to.to_vec());
        // Appending to the same series eases; different data jumps.
        assert!(is_append(&from, &to));
        assert!(!is_append(&from, &[(5.0, 0.0), (6.0, 1.0)]));
        assert!(!is_append(&from, &[(0.0, 0.0)]));
    }

    #[test]
    fn rebuilt_identical_data_never_restarts_easing() {
        let id: SharedString = "test-rebuilt".into();
        let key = || (1u64, vec![SharedString::from("a")]);
        let pts = |n: usize| Arc::new((0..n).map(|i| (i as f32, i as f32)).collect::<Vec<_>>());
        let t0 = std::time::Instant::now();
        assert!(ease_frame(&id, key(), &[pts(10)], t0, false).is_none());
        // Every render rebuilds the same points (like the PR chart): still nothing to animate.
        for _ in 0..3 {
            assert!(ease_frame(&id, key(), &[pts(10)], t0, false).is_none());
        }
        // An append eases, and the move ends.
        assert!(ease_frame(&id, key(), &[pts(11)], t0, false).is_some());
        let later = t0 + std::time::Duration::from_millis(300);
        assert!(ease_frame(&id, key(), &[pts(11)], later, false).is_none());
        // Another owner (a different run) with an "append-like" shape jumps.
        assert!(ease_frame(&id, (2, vec!["a".into()]), &[pts(12)], later, false).is_none());
        // Long histories jump too.
        let long = "test-long".into();
        assert!(ease_frame(&long, key(), &[pts(600)], t0, false).is_none());
        assert!(ease_frame(&long, key(), &[pts(601)], t0, false).is_none());
    }

    #[test]
    fn easing_survives_hidden_series_and_stops_under_reduce_motion() {
        let id: SharedString = "test-hide".into();
        let names =
            |n: usize| (7u64, (0..n).map(|i| SharedString::from(format!("s{i}"))).collect::<Vec<_>>());
        let pts = |n: usize| Arc::new((0..n).map(|i| (i as f32, i as f32)).collect::<Vec<_>>());
        let t0 = std::time::Instant::now();
        assert!(ease_frame(&id, names(2), &[pts(10), pts(10)], t0, false).is_none());
        // Hide one series, then restore both: appends ease again.
        assert!(ease_frame(&id, names(1), &[pts(10)], t0, false).is_none());
        assert!(ease_frame(&id, names(2), &[pts(10), pts(10)], t0, false).is_none());
        assert!(ease_frame(&id, names(2), &[pts(11), pts(11)], t0, false).is_some());
        // Reduce motion mid-move: the move stops at once.
        let mid = t0 + std::time::Duration::from_millis(100);
        assert!(ease_frame(&id, names(2), &[pts(11), pts(11)], mid, true).is_none());
        assert!(ease_frame(&id, names(2), &[pts(11), pts(11)], mid, false).is_none());
    }
}
