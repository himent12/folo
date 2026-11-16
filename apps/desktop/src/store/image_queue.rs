//! The order decoded images are fetched in: a few at a time, the newest request first.
//!
//! A grid flung through a big dataset asks for every tile it passes. Started all at once and in
//! order, those fetches held hundreds of threads in HTTP calls (and one more each for ureq's
//! timed address lookup) and the tiles left in view loaded last, seconds later (stress x4-soak-2).
//! Here at most [`LoadQueue::MAX_IN_FLIGHT`] run; the next one started is the newest asked for whose
//! slot is still loading, so a tile the cache evicted while it waited (scrolled far past) is
//! never fetched. "Newest" is the latest ask: a tile drawn again while it waits ([`LoadQueue::touch`],
//! a grid scrolled back) moves ahead of the ones queued after it.

use std::collections::{BTreeMap, HashMap};

use super::ImgKey;

pub struct LoadQueue<T> {
    /// Waiting fetches by when they were last asked for: the last entry is the newest.
    waiting: BTreeMap<u64, (ImgKey, T)>,
    /// Each waiting key's place in `waiting`.
    asked: HashMap<ImgKey, u64>,
    seq: u64,
    in_flight: usize,
    /// Every key started, in order.
    #[cfg(test)]
    pub started: Vec<ImgKey>,
}

impl<T> Default for LoadQueue<T> {
    fn default() -> Self {
        Self {
            waiting: BTreeMap::new(),
            asked: HashMap::new(),
            seq: 0,
            in_flight: 0,
            #[cfg(test)]
            started: Vec::new(),
        }
    }
}

impl<T> LoadQueue<T> {
    /// Fetches on their way at once (a browser's per-host connection count, about).
    pub const MAX_IN_FLIGHT: usize = 8;

    /// Queue a fetch for `key` (its slot already marked loading), as the newest; one already waiting
    /// for the same key is replaced.
    pub fn push(&mut self, key: ImgKey, job: T) {
        self.seq += 1;
        if let Some(old) = self.asked.insert(key.clone(), self.seq) {
            self.waiting.remove(&old);
        }
        self.waiting.insert(self.seq, (key, job));
    }

    /// `key` was asked for again (its tile drawn while still loading): if it is waiting, it becomes
    /// the newest. A fetch already on its way, or a key never queued, is left alone.
    pub fn touch(&mut self, key: &ImgKey) {
        let Some(at) = self.asked.get_mut(key) else { return };
        if *at == self.seq {
            return;
        }
        let Some(entry) = self.waiting.remove(&*at) else { return };
        self.seq += 1;
        *at = self.seq;
        self.waiting.insert(self.seq, entry);
    }

    /// The next fetch to start while fewer than [`Self::MAX_IN_FLIGHT`] are on their way: the newest
    /// whose key `wanted` still says yes to. The ones passed over are dropped.
    pub fn start_next(&mut self, mut wanted: impl FnMut(&ImgKey) -> bool) -> Option<(ImgKey, T)> {
        if self.in_flight >= Self::MAX_IN_FLIGHT {
            return None;
        }
        while let Some((_, (key, job))) = self.waiting.pop_last() {
            self.asked.remove(&key);
            if wanted(&key) {
                self.in_flight += 1;
                #[cfg(test)]
                self.started.push(key.clone());
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
        self.asked.clear();
        self.in_flight = 0;
    }

    #[cfg(test)]
    pub fn in_flight(&self) -> usize {
        self.in_flight
    }

    #[cfg(test)]
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

    #[test]
    fn a_waiting_key_asked_for_again_goes_next_and_is_started_once() {
        let mut q = LoadQueue::default();
        for i in 0..10 {
            q.push(key(i), i);
        }
        let first = q.start_next(|_| true).map(|(_, j)| j);
        assert_eq!(first, Some(9));
        q.touch(&key(3)); // scrolled back to tile 3
        q.touch(&key(9)); // on its way already: no second fetch
        q.touch(&key(42)); // never queued
        q.push(key(5), 50); // queued again: replaces the waiting one, as the newest
        assert_eq!(q.waiting(), 9);
        let rest: Vec<usize> = std::iter::from_fn(|| {
            q.done();
            q.start_next(|_| true).map(|(_, j)| j)
        })
        .collect();
        assert_eq!(rest, [50, 3, 8, 7, 6, 4, 2, 1, 0]);
        assert_eq!(q.started.iter().filter(|k| **k == key(9)).count(), 1);
    }
}
