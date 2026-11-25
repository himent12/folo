//! The paths `PathBuilder::build` made recently, by their input. Building is a pure function of the
//! builder (its contours, transform, dashes and options), and tessellating a chart's lines every frame was
//! about a quarter of a full frame of the live run page (lyon's stroker, at every redraw by the data pump).
//! A builder equal word for word to one built recently (every float compared by its bits) gets a copy of
//! that result, which is what building it again gives. Per thread, bounded in bytes, the least recently
//! used dropped first. Owner tag: `"style"`.

use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::{BuildHasher, Hasher};
use std::sync::Arc;

use crate::{BuiltPath, PathData, Pixels, Point, ScaledPixels};

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

/// Strong + weak refcount header of an `ArcInner` allocation.
const ARC_HEADER: usize = std::mem::size_of::<usize>() * 2;

/// The bytes an entry for a key of `key_words` and a path of `points` allocates (its key and its
/// `Arc<[Point<Pixels>]>` triangles, without spare capacity).
fn entry_bytes(key_words: usize, points: usize) -> usize {
    key_words * std::mem::size_of::<u32>() + ARC_HEADER + points * std::mem::size_of::<Point<Pixels>>()
}

/// The bytes a cached scaled `Arc<PathData>` allocates (its `ArcInner<PathData>` and triangle buffer).
fn scaled_bytes(points: usize) -> usize {
    ARC_HEADER + std::mem::size_of::<PathData>() + points * std::mem::size_of::<Point<ScaledPixels>>()
}

struct Entry {
    /// The key, without spare capacity.
    key: Box<[u32]>,
    /// The path, its triangles shared in an exact-sized `Arc<[Point<Pixels>]>`.
    path: BuiltPath<Pixels>,
    /// Device-pixel `PathData` cached for the last scale factor painted with this entry.
    scaled: Option<(u32, Arc<PathData>)>,
    last_used: u64,
}

impl Entry {
    /// What this entry's key and logical triangles allocate.
    fn base_bytes(&self) -> usize {
        entry_bytes(self.key.len(), self.path.triangles.len())
    }

    /// What this entry's cached scaled `PathData` allocates, if present.
    fn scaled_bytes(&self) -> usize {
        self.scaled.as_ref().map_or(0, |(_, data)| scaled_bytes(data.triangles.capacity()))
    }

    /// What this entry allocates: its key, triangles, and optional scaled `PathData`.
    fn bytes(&self) -> usize {
        self.base_bytes() + self.scaled_bytes()
    }
}

/// Built paths by the hash of their key (a collision keeps the newer one).
#[derive(Default)]
pub(crate) struct PathCache {
    entries: HashMap<u64, Entry>,
    /// What the entries allocate (keys, logical triangles, and cached scaled `PathData`); the table is
    /// counted by its capacity.
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
    /// What the cache holds: its entries' keys, triangles, cached scaled `PathData`, and its table at
    /// its allocated size.
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
    fn insert(&mut self, key: Vec<u32>, path: &mut BuiltPath<Pixels>) {
        // Measured before anything is copied: a path too big to keep costs no copy.
        let bytes = entry_bytes(key.len(), path.triangles.len());
        if bytes + SLOT > MAX_ENTRY {
            return;
        }
        self.clock += 1;
        let h = hash(&key);
        path.cache_hash = Some(h);
        // `Arc<[Point<Pixels>]>` has no spare capacity; the key is cut to its length.
        let entry =
            Entry { key: key.into_boxed_slice(), path: path.clone(), scaled: None, last_used: self.clock };
        if let Some(old) = self.entries.insert(h, entry) {
            self.heap -= old.bytes();
        }
        self.heap += bytes;
        if self.bytes() > BUDGET {
            self.evict();
        }
    }

    /// Returns cached device-pixel [`PathData`] for `path` at `scale_factor`, or scales and caches it
    /// on the matching entry (verified by `Arc::ptr_eq` on `path.triangles`) when it fits the budget.
    fn scaled_path_data(&mut self, path: &BuiltPath<Pixels>, scale_factor: f32) -> Arc<PathData> {
        let build_scaled = || {
            let scaled: Vec<Point<ScaledPixels>> =
                path.triangles.iter().map(|p| p.scale(scale_factor)).collect();
            Arc::new(PathData::from_triangles(scaled))
        };
        let Some(h) = path.cache_hash else {
            return build_scaled();
        };
        let scale_bits = scale_factor.to_bits();
        let Some(entry) = self.entries.get_mut(&h) else {
            return build_scaled();
        };
        if !Arc::ptr_eq(&entry.path.triangles, &path.triangles) {
            return build_scaled();
        }
        self.clock += 1;
        entry.last_used = self.clock;
        if let Some((bits, data)) = &entry.scaled
            && *bits == scale_bits
        {
            return Arc::clone(data);
        }
        let data = build_scaled();
        let extra = scaled_bytes(data.triangles.capacity());
        if entry.base_bytes() + extra + SLOT > MAX_ENTRY {
            return data;
        }
        let old_extra = entry.scaled_bytes();
        entry.scaled = Some((scale_bits, Arc::clone(&data)));
        self.heap = self.heap - old_extra + extra;
        if self.bytes() > BUDGET {
            self.evict();
        }
        data
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
    let mut path = build()?;
    let _ = CACHE.try_with(|c| c.borrow_mut().insert(key, &mut path));
    Ok(path)
}

/// Device-pixel [`PathData`] for `path` at `scale_factor`, reusing the thread-local cache's scaled
/// geometry when `path` came from it.
pub(crate) fn scaled_path_data(path: &BuiltPath<Pixels>, scale_factor: f32) -> Arc<PathData> {
    CACHE.try_with(|c| c.borrow_mut().scaled_path_data(path, scale_factor)).unwrap_or_else(|_| {
        let scaled: Vec<Point<ScaledPixels>> = path.triangles.iter().map(|p| p.scale(scale_factor)).collect();
        Arc::new(PathData::from_triangles(scaled))
    })
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
        BuiltPath { triangles: Arc::from(vec![p; 3 * n]), bounds: Default::default(), cache_hash: None }
    }

    /// A path of one triangle.
    fn tiny(i: u32) -> BuiltPath<Pixels> {
        triangles(i, 1)
    }

    /// What the cache counts is what it holds: a key with spare capacity (as `PathBuilder::key` may make
    /// one) is kept and counted at its length, the triangles in an exact-sized `Arc`, the table at its size.
    #[test]
    fn the_cache_counts_what_it_holds() {
        let mut c = PathCache::default();
        let mut key = Vec::with_capacity(100_000);
        key.extend([1, 2, 3]);
        let mut path = tiny(1);
        c.insert(key, &mut path);
        let e = c.entries.values().next().unwrap();
        assert_eq!(std::mem::size_of_val(&*e.key), 12, "no spare capacity is kept");
        assert_eq!(e.path.triangles.len(), 3);
        assert!(Arc::ptr_eq(&e.path.triangles, &path.triangles), "triangles are shared without copying");
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
            let mut path = tiny(i);
            c.insert(vec![i], &mut path);
            drop(path);
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
        let mut big = triangles(1, MAX_ENTRY / std::mem::size_of::<Point<Pixels>>());
        let key = vec![7];
        let before = counting::live();
        counting::reset_peak();
        c.insert(key, &mut big);
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
            c.insert(vec![i], &mut triangles(i, 100));
            i += 1;
        }
        let small_count = i;
        assert_eq!(c.evictions, 0);
        let mut big = triangles(0, MAX_ENTRY / std::mem::size_of::<Point<Pixels>>() / 6);
        c.insert(vec![u32::MAX], &mut big);
        assert_eq!(c.evictions, 1, "one pass");
        assert!(c.bytes() <= LOW_WATER, "{} bytes", c.bytes());
        assert_eq!(c.heap, c.entries.values().map(Entry::bytes).sum::<usize>());
        // The newest are kept, the oldest gone.
        assert!(c.get(&[u32::MAX]).is_some() && c.get(&[small_count - 1]).is_some());
        assert!(c.get(&[0]).is_none());
    }

    /// A cache full of small paths is cut down and then takes thousands of inserts before the next cut:
    /// the cut leaves room for the table's next growth.
    #[test]
    fn a_cache_of_small_paths_is_not_cut_on_every_insert() {
        let mut c = PathCache::default();
        let key = |i: u32| {
            let mut k = vec![0; 22];
            k[0] = i;
            k
        };
        assert_eq!(entry_bytes(22, 6), 136 + ARC_HEADER);
        let mut i = 0;
        while c.evictions == 0 {
            c.insert(key(i), &mut triangles(i, 2));
            i += 1;
        }
        for _ in 0..3000 {
            c.insert(key(i), &mut triangles(i, 2));
            i += 1;
            assert!(c.bytes() <= BUDGET, "{} bytes after {i}", c.bytes());
        }
        assert!(c.evictions <= 3, "{} cuts in 3000 inserts after the first", c.evictions - 1);
    }

    /// Repeated builds share `Arc<[Point<Pixels>]>`, repeated `scaled_path_data` calls at the same scale
    /// share `Arc<PathData>` (and its cached `geometry_hash`), replacing `triangles` bypasses stale reuse,
    /// and cached scaled entries stay within the byte budget.
    #[test]
    fn cached_path_shares_triangles_and_scaled_path_data_within_budget() {
        clear();
        let first = chart_line(100, 0., 1.5, false).build().unwrap();
        let second = chart_line(100, 0., 1.5, false).build().unwrap();
        assert!(Arc::ptr_eq(&first.triangles, &second.triangles), "cache hit shares triangles");

        let d1 = scaled_path_data(&first, 1.5);
        let d2 = scaled_path_data(&second, 1.5);
        assert!(Arc::ptr_eq(&d1, &d2), "same scale reuses Arc<PathData>");
        assert_eq!(d1.geometry_hash(), crate::raster::path::compute_geometry_hash(&d1));

        // Different scale rebuilds and replaces the cached scaled entry cleanly.
        let d3 = scaled_path_data(&second, 2.0);
        assert!(!Arc::ptr_eq(&d1, &d3));
        assert_eq!(d3.triangles.len(), first.triangles.len());
        let d4 = scaled_path_data(&first, 2.0);
        assert!(Arc::ptr_eq(&d3, &d4));

        // Replacing `triangles` on a caller-owned BuiltPath never returns stale cached geometry.
        let mut mutated = second;
        mutated.triangles = Arc::from([point(px(1.), px(2.)), point(px(3.), px(4.)), point(px(5.), px(6.))]);
        let dm = scaled_path_data(&mutated, 2.0);
        assert!(!Arc::ptr_eq(&d3, &dm));
        assert_eq!(dm.triangles.len(), 3);
    }

    /// Attaching scaled `PathData` tracks exact heap bytes (verified by the counting allocator), refuses
    /// to cache scaled geometry when base + scaled exceeds [`MAX_ENTRY`], and evicts oldest entries when
    /// attaching scaled geometry pushes total usage past [`BUDGET`].
    #[test]
    fn scaled_path_data_obeys_max_entry_budget_and_allocator_accounting() {
        // 1. Exact allocator accounting for scaled attachment and scale replacement.
        let base = counting::live();
        let mut c = PathCache::default();
        let mut p = triangles(1, 50);
        c.insert(vec![1], &mut p);
        let d1 = c.scaled_path_data(&p, 1.5);
        assert_eq!(c.heap, c.entries.values().map(Entry::bytes).sum::<usize>());
        let d2 = c.scaled_path_data(&p, 2.0);
        assert!(!Arc::ptr_eq(&d1, &d2));
        drop(d1);
        drop(d2);
        drop(p);
        let held = (counting::live() - base) as usize;
        assert!(held <= c.bytes(), "{held} bytes allocated, {} counted", c.bytes());
        assert!(c.bytes() - held <= 64, "{held} bytes allocated, {} counted", c.bytes());
        drop(c);
        assert_eq!(counting::live(), base, "all scaled and logical allocations freed");

        // 2. MAX_ENTRY guard: base fits (> MAX_ENTRY/2), but base + scaled > MAX_ENTRY.
        let mut c = PathCache::default();
        let pts_per_tri = 3 * std::mem::size_of::<Point<Pixels>>();
        let n_tris = (MAX_ENTRY * 3 / 4) / pts_per_tri;
        let mut large = triangles(7, n_tris);
        c.insert(vec![7], &mut large);
        assert_eq!(c.entries.len(), 1, "base entry fits under MAX_ENTRY");
        let heap_before = c.heap;
        let _scaled = c.scaled_path_data(&large, 1.5);
        assert!(
            c.entries.values().next().unwrap().scaled.is_none(),
            "scaled data exceeding MAX_ENTRY must not be retained"
        );
        assert_eq!(c.heap, heap_before);

        // 3. BUDGET eviction when attaching scaled data pushes total bytes over BUDGET.
        let mut c = PathCache::default();
        let mut kept = Vec::new();
        let mut i = 0u32;
        while c.bytes() + 32_768 < BUDGET {
            let mut path = triangles(i, 500);
            c.insert(vec![i], &mut path);
            kept.push(path);
            i += 1;
        }
        assert_eq!(c.evictions, 0);
        for path in &kept {
            let _ = c.scaled_path_data(path, 1.5);
            assert!(c.bytes() <= BUDGET, "{} bytes > BUDGET", c.bytes());
            assert_eq!(c.heap, c.entries.values().map(Entry::bytes).sum::<usize>());
        }
        assert!(c.evictions >= 1, "attaching scaled data near BUDGET must evict");
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
