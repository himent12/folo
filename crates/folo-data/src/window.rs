//! The mosaic window ([`crate::AugmentConfig::mosaic_window`]): mosaic and mixup partners drawn from the `K`
//! samples just before each one in the epoch's order, and a cache that keeps those samples decoded, so each image
//! of an epoch is decoded about once instead of four times.
//!
//! The partner of the sample at position `p` is the sample at position `(p − 1 − r) mod n` of the epoch's order,
//! `r` uniform in `0..min(K, n − 1)`: it depends only on `(seed, epoch, p)`, never on which worker got there first.
//! The cache is keyed by position. A slot holds what the loader's fetch returns, resized to long side `imgsz` (as
//! the RAM cache keeps it, so the transform gives the same result whether a slot was hit or decoded anew), and is
//! filled once by the first worker that needs it; the others wait for it instead of decoding it again. Slots
//! behind every sample still in flight (the collator's finished position less `K`) are dropped, so the cache
//! holds about `K` plus the loader's in-flight window of images. A partner that wraps around to the end of the
//! order (the first `K` samples of an epoch) is loaded directly and not kept.

use crate::augment::Fetched;
use crate::error::Result;
use crate::rng::Rng;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, OnceLock};

/// The position of a partner of position `pos` in an order of `len` samples, from the `k` positions before it
/// (wrapping around to the end). `len == 1` gives `pos` itself.
pub(crate) fn partner(pos: usize, len: usize, k: usize, rng: &mut Rng) -> usize {
    debug_assert!(pos < len && k > 0);
    if len == 1 {
        return pos;
    }
    let r = rng.below(k.min(len - 1));
    (pos + len - 1 - r) % len
}

/// A slot: the fetched sample, or `None` when its fetch failed (each caller then fetches again for its own
/// error).
type Slot = Arc<OnceLock<Option<Arc<Fetched>>>>;

/// Samples of the current epoch kept decoded by position (see the module docs).
#[derive(Default)]
pub(crate) struct WindowCache {
    slots: Mutex<BTreeMap<usize, Slot>>,
}

impl WindowCache {
    /// The sample at position `pos`: from its slot, or loaded with `load` (once, by the first caller) and kept.
    pub(crate) fn get(&self, pos: usize, load: impl Fn() -> Result<Fetched>) -> Result<Fetched> {
        let slot = {
            let mut slots = self.slots.lock().unwrap_or_else(|p| p.into_inner());
            slots.entry(pos).or_default().clone()
        };
        let mut failed = None;
        let got = slot.get_or_init(|| match load() {
            Ok(f) => Some(Arc::new(f)),
            Err(e) => {
                failed = Some(e);
                None
            }
        });
        match (got, failed) {
            (Some(f), _) => Ok(Fetched::clone(f)),
            (None, Some(e)) => Err(e),
            // Another worker's load failed; this one reports its own (deterministic) failure.
            (None, None) => load(),
        }
    }

    /// Drops the slots of positions below `pos`.
    pub(crate) fn evict_below(&self, pos: usize) {
        let mut slots = self.slots.lock().unwrap_or_else(|p| p.into_inner());
        *slots = slots.split_off(&pos);
    }

    /// Slots held (tests).
    #[cfg(test)]
    fn len(&self) -> usize {
        self.slots.lock().unwrap().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::DataError;
    use crate::image::{Image, Sample};
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn fetched(index: usize) -> Fetched {
        let image = Image { width: 2, height: 1, data: vec![index as u8; 6] };
        Fetched::decoded(index, Sample { image, boxes: Vec::new() })
    }

    /// Partners lie in the `k` positions before `pos` (wrapping), never at `pos`, and cover all of them.
    #[test]
    fn partners_come_from_the_previous_positions() {
        let mut rng = Rng::new(7);
        for (pos, len, k) in [(10, 50, 4), (2, 50, 4), (0, 50, 4), (5, 6, 100), (0, 1, 3)] {
            let mut seen = std::collections::BTreeSet::new();
            for _ in 0..400 {
                let q = partner(pos, len, k, &mut rng);
                assert!(q < len);
                seen.insert(q);
            }
            let want: std::collections::BTreeSet<usize> = if len == 1 {
                [0].into()
            } else {
                (1..=k.min(len - 1)).map(|d| (pos + len - d) % len).collect()
            };
            assert_eq!(seen, want, "pos {pos} len {len} k {k}");
        }
    }

    /// A slot loads once and serves copies; eviction drops the slots below a position; a failed load is not kept
    /// as a value and every caller gets an error.
    #[test]
    fn loads_once_and_evicts() {
        let cache = WindowCache::default();
        let loads = AtomicUsize::new(0);
        let load = |i: usize| {
            loads.fetch_add(1, Ordering::SeqCst);
            Ok(fetched(i))
        };
        for _ in 0..3 {
            for pos in 0..5 {
                assert_eq!(cache.get(pos, || load(pos)).unwrap().index, pos);
            }
        }
        assert_eq!(loads.load(Ordering::SeqCst), 5);
        cache.evict_below(3);
        assert_eq!(cache.len(), 2);
        cache.get(1, || load(1)).unwrap();
        assert_eq!(loads.load(Ordering::SeqCst), 6, "an evicted slot loads again");
        let bad = || Err(DataError::Format("unreadable".into()));
        assert!(cache.get(9, bad).is_err());
        assert!(cache.get(9, bad).is_err());
    }
}
