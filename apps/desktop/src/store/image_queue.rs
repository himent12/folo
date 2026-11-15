//! The order decoded images are fetched in: a few at a time, the newest request first.
//!
//! A grid flung through a big dataset asks for every tile it passes. Started all at once and in
//! order, those fetches held hundreds of threads in HTTP calls (and one more each for ureq's
//! timed address lookup) and the tiles left in view loaded last, seconds later (stress x4-soak-2).
//! Here at most [`LoadQueue::MAX_IN_FLIGHT`] run; the next one started is the newest asked for whose
//! slot is still loading, so a tile the cache evicted while it waited (scrolled far past) is
//! never fetched.

use super::ImgKey;

pub struct LoadQueue<T> {
    waiting: Vec<(ImgKey, T)>,
    in_flight: usize,
}

impl<T> Default for LoadQueue<T> {
    fn default() -> Self {
        Self { waiting: Vec::new(), in_flight: 0 }
    }
}

impl<T> LoadQueue<T> {
    /// Fetches on their way at once (a browser's per-host connection count, about).
    pub const MAX_IN_FLIGHT: usize = 8;

    /// Queue a fetch for `key` (its slot already marked loading).
    pub fn push(&mut self, key: ImgKey, job: T) {
        self.waiting.push((key, job));
    }

    /// The next fetch to start while fewer than [`Self::MAX_IN_FLIGHT`] are on their way: the newest
    /// whose key `wanted` still says yes to. The ones passed over are dropped.
    pub fn start_next(&mut self, mut wanted: impl FnMut(&ImgKey) -> bool) -> Option<(ImgKey, T)> {
        if self.in_flight >= Self::MAX_IN_FLIGHT {
            return None;
        }
        while let Some((key, job)) = self.waiting.pop() {
            if wanted(&key) {
                self.in_flight += 1;
                return Some((key, job));
            }
        }
        None
    }

    /// A started fetch answered.
    pub fn done(&mut self) {
        self.in_flight = self.in_flight.saturating_sub(1);
    }

    /// Forget everything (a new data source: what is on its way is dropped when it lands).
    pub fn clear(&mut self) {
        self.waiting.clear();
        self.in_flight = 0;
    }

    pub fn in_flight(&self) -> usize {
        self.in_flight
    }

    pub fn waiting(&self) -> usize {
        self.waiting.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::ImgKind;

    fn key(i: usize) -> ImgKey {
        ImgKey { dataset: "d".into(), image: i.to_string(), kind: ImgKind::Thumb }
    }

    #[test]
    fn starts_a_few_at_a_time_newest_first_and_skips_the_unwanted() {
        let mut q = LoadQueue::default();
        for i in 0..20 {
            q.push(key(i), i);
        }
        let started: Vec<usize> = std::iter::from_fn(|| q.start_next(|_| true).map(|(_, j)| j)).collect();
        assert_eq!(started, (12..20).rev().collect::<Vec<_>>());
        assert_eq!((q.in_flight(), q.waiting()), (LoadQueue::<usize>::MAX_IN_FLIGHT, 12));
        // One answers; the odd keys were evicted meanwhile: the newest even one goes next.
        q.done();
        assert_eq!(q.start_next(|k| k.image.parse::<usize>().unwrap() % 2 == 0).map(|(_, j)| j), Some(10));
        assert_eq!(q.waiting(), 10, "11 was dropped on the way");
        q.clear();
        assert_eq!((q.in_flight(), q.waiting()), (0, 0));
        assert!(q.start_next(|_| true).is_none());
    }
}
