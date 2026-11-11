//! Merging a reloaded list with what the stream said since it was asked for, and placing one item.

use std::collections::HashMap;

use crate::api::*;

/// Installs a reloaded list over the one on screen. A snapshot can be older than the events applied
/// since it was asked for (count `since`): an item the stream changed after that keeps the stream's
/// version (unless the snapshot has it ended and the stream doesn't: an end is final), one the stream
/// removed stays removed, and one the stream added stays. Returns the list and whether it disagrees
/// with the snapshot in a way that can change the order (another reload then settles it).
#[allow(clippy::too_many_arguments)]
pub fn merge_snapshot<T: Clone>(
    current: Option<&[T]>,
    fresh: Vec<T>,
    since: u64,
    touched: &HashMap<String, u64>,
    removed: &HashMap<String, u64>,
    id: impl Fn(&T) -> &str,
    ended: impl Fn(&T) -> bool,
    place: impl Fn(&mut Vec<T>, T),
) -> (Vec<T>, bool) {
    let newer = |i: &str| touched.get(i).is_some_and(|&s| s > since);
    let gone = |i: &str| removed.get(i).is_some_and(|&s| s > since);
    let mut differs = false;
    let mut out: Vec<T> = Vec::with_capacity(fresh.len());
    for f in fresh {
        if gone(id(&f)) {
            continue;
        }
        let cur = current.and_then(|c| c.iter().find(|x| id(x) == id(&f)));
        match cur {
            Some(c) if newer(id(&f)) && !(ended(&f) && !ended(c)) => {
                differs |= ended(c) != ended(&f);
                out.push(c.clone());
            }
            _ => out.push(f),
        }
    }
    for c in current.into_iter().flatten() {
        if newer(id(c)) && !gone(id(c)) && !out.iter().any(|o| id(o) == id(c)) {
            differs |= ended(c);
            place(&mut out, c.clone());
        }
    }
    (out, differs)
}

/// Inserts `e` where it belongs in a leaderboard that already has the rest, or replaces its old
/// version in place. A new entry goes first among the unfinished ones (the server lists those
/// newest first, after every finished one).
pub fn upsert_entry(list: &mut Vec<BenchEntry>, e: BenchEntry) {
    if let Some(old) = list.iter_mut().find(|x| x.id == e.id) {
        // An entry that has ended never runs again: an older answer can't bring it back.
        if !old.status.is_active() && e.status.is_active() {
            return;
        }
        *old = e;
    } else {
        let at = list.iter().position(|x| x.status != BenchEntryStatus::Done).unwrap_or(list.len());
        list.insert(at, e);
    }
}

/// Inserts `l` first (newest first), or replaces its old version in place.
pub fn upsert_loop(list: &mut Vec<BenchLoop>, l: BenchLoop) {
    if let Some(old) = list.iter_mut().find(|x| x.id == l.id) {
        if !old.status.is_active() && l.status.is_active() {
            return;
        }
        *old = l;
    } else {
        list.insert(0, l);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(id: &str, status: BenchEntryStatus) -> BenchEntry {
        BenchEntry {
            id: id.into(),
            bench_id: "bn_1".into(),
            name: id.into(),
            contestant: BenchContestant::External,
            status,
            created: "2026-11-10T00:00:00.000Z".into(),
            finished: None,
            progress: BenchProgress::default(),
            images: 10,
            conditions: true,
            scored: true,
            fingerprint: "f".into(),
            missing_classes: vec![],
            score: None,
            map50_95: None,
            ci95: None,
            conditions_run: 0,
            speed: None,
            usage: None,
            error: None,
            hint: None,
        }
    }

    /// `merge_snapshot` for entries, as the leaderboard's reload calls it.
    fn merge(
        current: &[BenchEntry],
        fresh: Vec<BenchEntry>,
        since: u64,
        touched: &[(&str, u64)],
        removed: &[(&str, u64)],
    ) -> (Vec<BenchEntry>, bool) {
        let map = |v: &[(&str, u64)]| v.iter().map(|(k, s)| (k.to_string(), *s)).collect::<HashMap<_, _>>();
        merge_snapshot(
            Some(current),
            fresh,
            since,
            &map(touched),
            &map(removed),
            |e: &BenchEntry| e.id.as_str(),
            |e| !e.status.is_active(),
            upsert_entry,
        )
    }

    #[test]
    fn an_older_snapshot_never_undoes_what_the_stream_said() {
        use BenchEntryStatus::*;
        // Asked at event 3; the stream then said `b` finished (event 4). The snapshot still has it
        // running: the end stays, and the order is reloaded.
        let current = vec![entry("a", Done), entry("b", Done)];
        let (list, again) = merge(&current, vec![entry("a", Done), entry("b", Running)], 3, &[("b", 4)], &[]);
        assert_eq!(list[1].status, Done);
        assert!(again);
        // Progress the stream reported later is kept too, without another reload (same order).
        let mut ahead = entry("c", Running);
        ahead.progress.done = 9;
        let (list, again) = merge(&[ahead], vec![entry("c", Running)], 3, &[("c", 5)], &[]);
        assert_eq!((list[0].progress.done, again), (9, false));
        // A snapshot that has an end the stream hasn't sent yet wins: ends are final.
        let (list, _) = merge(&[entry("d", Running)], vec![entry("d", Failed)], 3, &[("d", 4)], &[]);
        assert_eq!(list[0].status, Failed);
        // Removed after the request: it stays gone. Added by the stream after it: it stays.
        let (list, _) = merge(
            &[entry("e", Queued)],
            vec![entry("a", Done), entry("gone", Done)],
            3,
            &[("e", 6), ("gone", 5)],
            &[("gone", 5)],
        );
        let ids: Vec<&str> = list.iter().map(|e| e.id.as_str()).collect();
        assert_eq!(ids, ["a", "e"]);
        // Nothing said since the request: the snapshot is the list.
        let (list, again) = merge(&[entry("x", Running)], vec![entry("y", Done)], 9, &[("x", 4)], &[]);
        assert_eq!((list.len(), list[0].id.as_str(), again), (1, "y", false));
    }

    #[test]
    fn an_ended_entry_or_loop_never_runs_again() {
        let mut list = vec![entry("a", BenchEntryStatus::Done)];
        upsert_entry(&mut list, entry("a", BenchEntryStatus::Running));
        assert_eq!(list[0].status, BenchEntryStatus::Done);
    }

    #[test]
    fn new_entries_go_first_among_the_unfinished() {
        use BenchEntryStatus::*;
        let mut list = vec![entry("a", Done), entry("b", Done), entry("c", Failed)];
        upsert_entry(&mut list, entry("d", Queued));
        let ids: Vec<_> = list.iter().map(|e| e.id.as_str()).collect();
        assert_eq!(ids, ["a", "b", "d", "c"]);
        // An update replaces in place, whatever its status now.
        upsert_entry(&mut list, entry("d", Running));
        assert_eq!(list[2].status, Running);
        assert_eq!(list.len(), 4);
    }
}
