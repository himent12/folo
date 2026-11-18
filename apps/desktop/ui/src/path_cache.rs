//! The paths `PathBuilder::build` made recently, by their input. Building is a pure function of the
//! builder (its contours, transform, dashes and options), and tessellating a chart's lines every frame was
//! about a quarter of a full frame of the live run page (lyon's stroker, at every redraw by the data pump).
//! A builder equal word for word to one built recently (every float compared by its bits) gets a copy of
//! that result, which is what building it again gives. Per thread, bounded in bytes, the least recently
//! used dropped first. Owner tag: `"style"`.

use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::{BuildHasher, Hasher};

use crate::{BuiltPath, Pixels};

/// Bytes the cache may hold (keys and triangles): a few hundred chart lines.
const BUDGET: usize = 8 << 20;
/// A path whose key or triangles alone are bigger is never kept.
const MAX_ENTRY: usize = BUDGET / 8;

struct Entry {
    key: Vec<u32>,
    path: BuiltPath<Pixels>,
    last_used: u64,
}

impl Entry {
    fn bytes(&self) -> usize {
        entry_bytes(&self.key, &self.path)
    }
}

fn entry_bytes(key: &[u32], path: &BuiltPath<Pixels>) -> usize {
    key.len() * 4 + std::mem::size_of_val(path.triangles.as_slice()) + std::mem::size_of::<Entry>()
}

/// Built paths by the hash of their key (a collision keeps the newer one).
#[derive(Default)]
pub(crate) struct PathCache {
    entries: HashMap<u64, Entry>,
    bytes: usize,
    clock: u64,
    hits: u64,
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
    /// The path built from `key` before, if it is still here.
    fn get(&mut self, key: &[u32]) -> Option<BuiltPath<Pixels>> {
        self.clock += 1;
        let entry = self.entries.get_mut(&hash(key)).filter(|e| e.key == key)?;
        entry.last_used = self.clock;
        self.hits += 1;
        Some(entry.path.clone())
    }

    /// Keeps `path` as what `key` builds, dropping the least recently used paths beyond the budget.
    fn insert(&mut self, key: Vec<u32>, path: &BuiltPath<Pixels>) {
        let bytes = entry_bytes(&key, path);
        if bytes > MAX_ENTRY {
            return;
        }
        self.clock += 1;
        let entry = Entry { key, path: path.clone(), last_used: self.clock };
        if let Some(old) = self.entries.insert(hash(&entry.key), entry) {
            self.bytes -= old.bytes();
        }
        self.bytes += bytes;
        while self.bytes > BUDGET {
            let Some((&h, _)) = self.entries.iter().min_by_key(|(_, e)| e.last_used) else { break };
            if let Some(old) = self.entries.remove(&h) {
                self.bytes -= old.bytes();
            }
        }
    }
}

/// `build(key)`'s result, from this thread's cache when `key` was built recently; a result that is an
/// error is never kept.
pub(crate) fn cached<E>(
    key: Vec<u32>,
    build: impl FnOnce() -> Result<BuiltPath<Pixels>, E>,
) -> Result<BuiltPath<Pixels>, E> {
    if let Some(hit) = CACHE.with_borrow_mut(|c| c.get(&key)) {
        return Ok(hit);
    }
    let path = build()?;
    CACHE.with_borrow_mut(|c| c.insert(key, &path));
    Ok(path)
}

/// This thread's cache: (entries, bytes, hits so far). For tests.
#[cfg(test)]
pub(crate) fn stats() -> (usize, usize, u64) {
    CACHE.with_borrow(|c| (c.entries.len(), c.bytes, c.hits))
}

/// Empties this thread's cache. For tests.
#[cfg(test)]
pub(crate) fn clear() {
    CACHE.with_borrow_mut(|c| *c = PathCache::default());
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
