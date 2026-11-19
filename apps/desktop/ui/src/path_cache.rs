//! The paths `PathBuilder::build` made recently, by their input. Building is a pure function of the
//! builder (its contours, transform, dashes and options), and tessellating a chart's lines every frame was
//! about a quarter of a full frame of the live run page (lyon's stroker, at every redraw by the data pump).
//! A builder equal word for word to one built recently (every float compared by its bits) gets a copy of
//! that result, which is what building it again gives. Per thread, bounded in bytes, the least recently
//! used dropped first. Owner tag: `"style"`.

use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::{BuildHasher, Hasher};

use crate::{BuiltPath, Pixels, Point};

/// Bytes the cache may hold after any call: its keys, its triangles and its table at the size the table
/// has allocated (not only its entries). A few hundred chart lines.
const BUDGET: usize = 8 << 20;
/// What a cache over [`BUDGET`] is cut down to, the least recently used first, in one pass: the next
/// inserts then fit without another.
const LOW_WATER: usize = BUDGET / 4 * 3;
/// A path whose entry alone is bigger is never kept.
const MAX_ENTRY: usize = BUDGET / 8;

/// One slot of the table: the hash and the entry, and the slot's control byte.
const SLOT: usize = std::mem::size_of::<(u64, Entry)>() + 1;

/// The bytes of the table a `HashMap` of entries allocates to hold `n` of them: the size of its table when
/// `capacity()` is `n`, and after `shrink_to_fit` with `n` entries. Std's table (hashbrown) has a power of
/// two of slots (at least 4), at most seven in eight of them used, and one group of control bytes (16)
/// beyond them; a table that grows doubles.
fn table_bytes(n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    let slots = match n {
        0..4 => 4,
        4..8 => 8,
        _ => (n * 8 / 7).next_power_of_two(),
    };
    slots * SLOT + 16
}

/// The bytes an entry for a key of `key_words` and a path of `points` allocates (its key and its
/// triangles, without spare capacity).
fn entry_bytes(key_words: usize, points: usize) -> usize {
    key_words * std::mem::size_of::<u32>() + points * std::mem::size_of::<Point<Pixels>>()
}

struct Entry {
    /// The key, without spare capacity.
    key: Box<[u32]>,
    /// The path, its triangles without spare capacity.
    path: BuiltPath<Pixels>,
    last_used: u64,
}

impl Entry {
    /// What this entry allocates: its key and its triangles (its slot is the table's).
    fn bytes(&self) -> usize {
        entry_bytes(self.key.len(), self.path.triangles.capacity())
    }
}

/// Built paths by the hash of their key (a collision keeps the newer one).
#[derive(Default)]
pub(crate) struct PathCache {
    entries: HashMap<u64, Entry>,
    /// What the entries allocate (keys and triangles); the table is counted by its capacity.
    heap: usize,
    clock: u64,
    hits: u64,
    /// Times the cache went over [`BUDGET`] and was cut down.
    evictions: u64,
}

thread_local! {
    static CACHE: RefCell<PathCache> = RefCell::new(PathCache::default());
}

/// A fixed hasher (FxHash's step): the key is already a list of words, and the map needs no
/// protection from chosen collisions (a collision is a miss, compared word for word).
fn hash(key: &[u32]) -> u64 {
    let mut h = std::hash::BuildHasherDefault::<WordHasher>::default().build_hasher();
    for w in key {
        h.write_u32(*w);
    }
    h.finish()
}

#[derive(Default)]
struct WordHasher(u64);

impl Hasher for WordHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for b in bytes {
            self.write_u32(u32::from(*b));
        }
    }

    fn write_u32(&mut self, w: u32) {
        self.0 = (self.0.rotate_left(5) ^ u64::from(w)).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
}

impl PathCache {
    /// What the cache holds: its entries' keys and triangles, and its table at its allocated size.
    fn bytes(&self) -> usize {
        self.heap + table_bytes(self.entries.capacity())
    }

    /// The path built from `key` before, if it is still here.
    fn get(&mut self, key: &[u32]) -> Option<BuiltPath<Pixels>> {
        self.clock += 1;
        let entry = self.entries.get_mut(&hash(key)).filter(|e| *e.key == *key)?;
        entry.last_used = self.clock;
        self.hits += 1;
        Some(entry.path.clone())
    }

    /// Keeps `path` as what `key` builds, dropping the least recently used paths beyond the budget.
    fn insert(&mut self, key: Vec<u32>, path: &BuiltPath<Pixels>) {
        // Measured before anything is copied: a path too big to keep costs no copy.
        let bytes = entry_bytes(key.len(), path.triangles.len());
        if bytes + SLOT > MAX_ENTRY {
            return;
        }
        self.clock += 1;
        // A clone's triangles have no spare capacity; the key is cut to its length.
        let entry = Entry { key: key.into_boxed_slice(), path: path.clone(), last_used: self.clock };
        if let Some(old) = self.entries.insert(hash(&entry.key), entry) {
            self.heap -= old.bytes();
        }
        self.heap += bytes;
        if self.bytes() > BUDGET {
            self.evict();
        }
    }

    /// Drops the least recently used paths until the cache, its table cut to what is left, is down to
    /// [`LOW_WATER`]: one sort of the entries by age, however many go. The table keeps its size when
    /// entries go, so it is then rebuilt for the rest (a table grown for many tiny paths was most of what
    /// the cache held). The table counted is the one for a quarter more entries than are left: cut to a
    /// table exactly full, the next insert would double it, go over [`BUDGET`] again and evict again, on
    /// every insert or every second one (a cache of small paths, about 130 bytes each).
    fn evict(&mut self) {
        self.evictions += 1;
        let mut by_age: Vec<(u64, u64)> = self.entries.iter().map(|(&h, e)| (e.last_used, h)).collect();
        by_age.sort_unstable();
        let mut left = self.entries.len();
        for (_, h) in by_age {
            if self.heap + table_bytes(left + left / 4 + 1) <= LOW_WATER {
                break;
            }
            if let Some(old) = self.entries.remove(&h) {
                self.heap -= old.bytes();
                left -= 1;
            }
        }
        self.entries.shrink_to_fit();
    }
}

/// `build(key)`'s result, from this thread's cache when `key` was built recently; a result that is an
/// error is never kept. While the thread is torn down and its cache already gone (a path built by another
/// thread local's destructor), the path is built without the cache.
pub(crate) fn cached<E>(
    key: Vec<u32>,
    build: impl FnOnce() -> Result<BuiltPath<Pixels>, E>,
) -> Result<BuiltPath<Pixels>, E> {
    if let Some(hit) = CACHE.try_with(|c| c.borrow_mut().get(&key)).ok().flatten() {
        return Ok(hit);
    }
    let path = build()?;
    let _ = CACHE.try_with(|c| c.borrow_mut().insert(key, &path));
    Ok(path)
}

/// This thread's cache: (entries, bytes, hits so far). For tests.
#[cfg(test)]
pub(crate) fn stats() -> (usize, usize, u64) {
    CACHE.with_borrow(|c| (c.entries.len(), c.bytes(), c.hits))
}

/// Empties this thread's cache. For tests.
#[cfg(test)]
pub(crate) fn clear() {
    CACHE.with_borrow_mut(|c| *c = PathCache::default());
}

/// The allocator of this crate's unit tests: the system's, counting per thread the bytes allocated and not
/// freed (the sizes asked for), so a test can measure what a structure it built holds.
#[cfg(test)]
pub(crate) mod counting {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;

    thread_local! {
        /// Bytes this thread allocated and has not freed.
        static LIVE: Cell<isize> = const { Cell::new(0) };
        /// The most `LIVE` was since [`reset_peak`].
        static PEAK: Cell<isize> = const { Cell::new(0) };
    }

    fn add(bytes: isize) {
        // `try_with`: an allocation while the thread's locals are being torn down goes uncounted.
        let _ = LIVE.try_with(|live| {
            live.set(live.get() + bytes);
            let _ = PEAK.try_with(|peak| peak.set(peak.get().max(live.get())));
        });
    }

    /// This thread's live bytes.
    pub fn live() -> isize {
        LIVE.with(Cell::get)
    }

    /// This thread's most live bytes since [`reset_peak`].
    pub fn peak() -> isize {
        PEAK.with(Cell::get)
    }

    /// Starts this thread's peak again from its live bytes.
    pub fn reset_peak() {
        PEAK.with(|peak| peak.set(live()));
    }

    struct Counting;

    #[global_allocator]
    static ALLOC: Counting = Counting;

    // SAFETY: every method forwards to `System` with the caller's arguments unchanged; the counters are
    // const-initialised thread locals without destructors, which allocate nothing.
    unsafe impl GlobalAlloc for Counting {
        // Counted only when the allocation succeeded: a null result holds nothing.
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            // SAFETY: the caller upholds `GlobalAlloc::alloc`'s contract, which `System` shares.
            let p = unsafe { System.alloc(layout) };
            if !p.is_null() {
                add(layout.size() as isize);
            }
            p
        }

        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            // SAFETY: as `alloc`.
            let p = unsafe { System.alloc_zeroed(layout) };
            if !p.is_null() {
                add(layout.size() as isize);
            }
            p
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            add(-(layout.size() as isize));
            // SAFETY: `ptr` came from this allocator, so from `System`, with `layout`.
            unsafe { System.dealloc(ptr, layout) }
        }

        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            // SAFETY: as `dealloc`, and the caller upholds `realloc`'s size contract.
            let p = unsafe { System.realloc(ptr, layout, new_size) };
            // Only when it succeeded (on failure the old block stays, as counted). The new block is counted
            // before the old one goes, as both may be held during the copy.
            if !p.is_null() {
                add(new_size as isize);
                add(-(layout.size() as isize));
            }
            p
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PathBuilder, point, px};

    fn chart_line(n: usize, wiggle: f32, width: f32, dashed: bool) -> PathBuilder {
        let mut b = PathBuilder::stroke(px(width));
        if dashed {
            b = b.dash_array(&[px(4.), px(3.)]);
        }
        b.move_to(point(px(0.), px(50.)));
        for i in 1..n {
            let x = i as f32 * 3.;
            b.line_to(point(px(x), px(50. + (x * 0.37).sin() * 20. + wiggle)));
        }
        b
    }

    #[test]
    fn a_path_built_again_is_the_same_path_and_comes_from_the_cache() {
        clear();
        let fresh = chart_line(200, 0., 1.5, false).build_uncached().unwrap();
        let first = chart_line(200, 0., 1.5, false).build().unwrap();
        let (_, _, hits) = stats();
        let again = chart_line(200, 0., 1.5, false).build().unwrap();
        assert_eq!(first, fresh);
        assert_eq!(again, fresh);
        assert_eq!(stats().2, hits + 1, "the second build is a hit");
    }

    #[test]
    fn any_difference_in_the_input_builds_anew() {
        clear();
        let base = chart_line(120, 0., 1.5, false);
        let _ = base.clone().build().unwrap();
        let variants = [
            chart_line(120, 0.001, 1.5, false), // a point moved by a hair
            chart_line(121, 0., 1.5, false),    // one more point
            chart_line(120, 0., 2.0, false),    // another width
            chart_line(120, 0., 1.5, true),     // dashed
            {
                let mut b = base.clone();
                b.close();
                b
            },
            {
                let mut b = base.clone();
                b.translate(point(px(0.5), px(0.)));
                b
            },
            {
                let mut fill = PathBuilder::fill();
                fill.add_polygon(
                    &[point(px(0.), px(0.)), point(px(9.), px(0.)), point(px(0.), px(9.))],
                    true,
                );
                fill
            },
        ];
        for v in variants {
            let (_, _, hits) = stats();
            let got = v.clone().build().unwrap();
            assert_eq!(stats().2, hits, "{v:?} is not the cached path");
            assert_eq!(got, v.build_uncached().unwrap());
        }
        // Negative zero is another float's bits: built anew, and the same triangles either way.
        let mut zero = PathBuilder::fill();
        zero.add_polygon(&[point(px(-0.), px(0.)), point(px(9.), px(0.)), point(px(0.), px(9.))], true);
        let (_, _, hits) = stats();
        let _ = zero.build().unwrap();
        assert_eq!(stats().2, hits);
    }

    /// A path of `n` triangles at `i`, as the cache holds it.
    fn triangles(i: u32, n: usize) -> BuiltPath<Pixels> {
        let p = point(px(i as f32), px(0.));
        BuiltPath { triangles: vec![p; 3 * n], bounds: Default::default() }
    }

    /// A path of one triangle.
    fn tiny(i: u32) -> BuiltPath<Pixels> {
        triangles(i, 1)
    }

    /// What the cache counts is what it holds: a key with spare capacity (as `PathBuilder::key` may make
    /// one) is kept and counted at its length, the triangles at their capacity, the table at its size.
    #[test]
    fn the_cache_counts_what_it_holds() {
        let mut c = PathCache::default();
        let mut key = Vec::with_capacity(100_000);
        key.extend([1, 2, 3]);
        let mut path = tiny(1);
        path.triangles.reserve(10_000);
        c.insert(key, &path);
        let e = c.entries.values().next().unwrap();
        assert_eq!(std::mem::size_of_val(&*e.key), 12, "no spare capacity is kept");
        assert_eq!(e.path.triangles.capacity(), 3);
        assert_eq!(c.heap, e.bytes());
        assert_eq!(c.bytes(), e.bytes() + table_bytes(c.entries.capacity()));
        assert!(c.bytes() < 1000, "{} bytes", c.bytes());
    }

    /// Measured by the allocator: filled with many tiny paths (where the table is most of what the cache
    /// holds) and cut down again and again, the cache never allocates more than it counts, nor more than
    /// [`BUDGET`]; its count is no more than a little over what it allocated.
    #[test]
    fn the_cache_allocates_no_more_than_its_budget() {
        let base = counting::live();
        let mut c = PathCache::default();
        let mut most = 0;
        for i in 0..300_000 {
            c.insert(vec![i], &tiny(i));
            let held = (counting::live() - base) as usize;
            assert!(held <= c.bytes(), "{held} bytes allocated, {} counted, after {i}", c.bytes());
            assert!(c.bytes() <= BUDGET, "{} bytes after {i}", c.bytes());
            assert!(c.bytes() - held <= 64, "{held} bytes allocated, {} counted, after {i}", c.bytes());
            most = most.max(held);
        }
        assert!(c.evictions >= 2, "{} evictions", c.evictions);
        assert!(most > BUDGET / 2, "the budget was used: {most} bytes at most");
        drop(c);
        assert_eq!(counting::live(), base, "all of it freed");
    }

    /// A path over the per-entry limit is turned away before it is copied: it costs no copy of its
    /// triangles, however big.
    #[test]
    fn a_path_too_big_to_keep_is_not_copied() {
        let mut c = PathCache::default();
        let big = triangles(1, MAX_ENTRY / std::mem::size_of::<Point<Pixels>>());
        let key = vec![7];
        let before = counting::live();
        counting::reset_peak();
        c.insert(key, &big);
        assert!(c.entries.is_empty());
        let peak = (counting::peak() - before) as usize;
        assert!(peak < 1024, "{peak} bytes allocated on the way");
    }

    /// One big path into a cache full of small ones drops all it must in one pass (a scan of the whole
    /// map per dropped entry made that thousands of scans in one frame), oldest first.
    #[test]
    fn a_big_path_into_a_full_cache_evicts_in_one_pass() {
        let mut c = PathCache::default();
        let mut i = 0;
        while c.bytes() + 4096 < BUDGET {
            c.insert(vec![i], &triangles(i, 100));
            i += 1;
        }
        let small_count = i;
        assert_eq!(c.evictions, 0);
        let big = triangles(0, MAX_ENTRY / std::mem::size_of::<Point<Pixels>>() / 6);
        c.insert(vec![u32::MAX], &big);
        assert_eq!(c.evictions, 1, "one pass");
        assert!(c.bytes() <= LOW_WATER, "{} bytes", c.bytes());
        assert_eq!(c.heap, c.entries.values().map(Entry::bytes).sum::<usize>());
        // The newest are kept, the oldest gone.
        assert!(c.get(&[u32::MAX]).is_some() && c.get(&[small_count - 1]).is_some());
        assert!(c.get(&[0]).is_none());
    }

    /// A cache full of small paths (136 bytes each, as a filled four-point polygon's) is cut down and then
    /// takes thousands of inserts before the next cut: the cut leaves room for the table's next growth.
    /// Cut to a table exactly full, the next insert doubled the table, went over the budget and cut again
    /// (a sort of some 28 000 entries) on every insert or every second one.
    #[test]
    fn a_cache_of_small_paths_is_not_cut_on_every_insert() {
        let mut c = PathCache::default();
        let key = |i: u32| {
            let mut k = vec![0; 22];
            k[0] = i;
            k
        };
        assert_eq!(entry_bytes(22, 6), 136);
        let mut i = 0;
        while c.evictions == 0 {
            c.insert(key(i), &triangles(i, 2));
            i += 1;
        }
        for _ in 0..3000 {
            c.insert(key(i), &triangles(i, 2));
            i += 1;
            assert!(c.bytes() <= BUDGET, "{} bytes after {i}", c.bytes());
        }
        assert!(c.evictions <= 3, "{} cuts in 3000 inserts after the first", c.evictions - 1);
    }

    /// A path built while the thread is torn down, by the destructor of a thread local destroyed after
    /// the cache: built without the cache (reaching it there panicked, and a panic in a destructor aborts).
    #[test]
    fn a_path_built_after_the_cache_is_gone_is_built_uncached() {
        struct BuildsOnDrop(std::sync::mpsc::Sender<(bool, bool)>);
        impl Drop for BuildsOnDrop {
            fn drop(&mut self) {
                let gone = CACHE.try_with(|_| ()).is_err();
                let built = std::panic::catch_unwind(|| chart_line(50, 0., 1.5, false).build().is_ok());
                let _ = self.0.send((gone, built.unwrap_or(false)));
            }
        }
        thread_local! {
            static LATE: RefCell<Option<BuildsOnDrop>> = const { RefCell::new(None) };
        }
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            // Set before the cache is first used: thread locals are destroyed in the reverse order.
            LATE.with_borrow_mut(|l| *l = Some(BuildsOnDrop(tx)));
            let _ = chart_line(50, 0., 1.5, false).build().unwrap();
        })
        .join()
        .unwrap();
        let (gone, built) = rx.recv().unwrap();
        assert!(gone, "the cache was still there: not the case this test is for");
        assert!(built, "building the path panicked");
    }

    #[test]
    fn the_cache_stays_within_its_budget() {
        clear();
        // Long lines (about 120 KB of triangles each), many more than fit.
        for i in 0..200 {
            let _ = chart_line(800, i as f32, 1.5, false).build().unwrap();
            assert!(stats().1 <= BUDGET, "{} bytes after {i}", stats().1);
        }
        let (entries, bytes, _) = stats();
        assert!(entries < 200 && bytes > BUDGET / 2, "{entries} entries, {bytes} bytes");
        // The newest is still there, the oldest is gone.
        let (_, _, hits) = stats();
        let _ = chart_line(800, 199., 1.5, false).build().unwrap();
        assert_eq!(stats().2, hits + 1);
        let _ = chart_line(800, 0., 1.5, false).build().unwrap();
        assert_eq!(stats().2, hits + 1);
    }
}
