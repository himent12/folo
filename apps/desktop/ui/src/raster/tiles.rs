//! Tiles: 64x64 device pixels. Items are binned by their pixel rectangle, each tile's ordered list is
//! cut at the topmost item that covers it with an opaque colour, the rest is hashed, and the tiles
//! whose hash changed are the frame's damage.

use super::geom::IRect;
use super::hash::{finish, mix};
use super::items::Item;
use super::pool::{Pool, for_each};
use std::sync::Mutex;
use crate::{Bounds, DevicePixels, point, size};

/// The tile edge in device pixels.
pub const TILE: i32 = 64;

fn tile_rect(t: usize, cols: i32, width: i32, height: i32) -> IRect {
    let (tx, ty) = (t as i32 % cols, t as i32 / cols);
    IRect::new(tx * TILE, ty * TILE, ((tx + 1) * TILE).min(width), ((ty + 1) * TILE).min(height))
}

/// The list's base pointer, shared by the threads that fill disjoint ranges of it.
struct ListPtr(*mut u32);

// SAFETY: used only by `Tiles::bin`, whose threads write disjoint slots (see there).
unsafe impl Sync for ListPtr {}

/// A frame's tile grid and what each tile holds.
#[derive(Default)]
pub(crate) struct Tiles {
    pub cols: i32,
    pub rows: i32,
    width: i32,
    height: i32,
    /// Last frame's hash of each tile (`None`: unknown, repaint).
    hashes: Vec<Option<u64>>,
    /// `offsets[t]..offsets[t + 1]` are tile `t`'s entries in `list`.
    offsets: Vec<u32>,
    cursor: Vec<u32>,
    /// Per part and tile, how many entries the part adds (binning scratch).
    counts: Vec<u32>,
    /// Item indices, per tile, in painter's order.
    list: Vec<u32>,
    /// The first entry of each tile that is drawn (entries before it are hidden).
    start: Vec<u32>,
    /// Whether that first entry covers the tile opaquely (so the tile needs no clear).
    occluded: Vec<bool>,
}

impl Tiles {
    /// Sizes the grid for a `width` x `height` frame; forgets every hash when the size changes.
    pub fn resize(&mut self, width: u32, height: u32) -> bool {
        let (w, h) = (width as i32, height as i32);
        if w == self.width && h == self.height && !self.hashes.is_empty() {
            return false;
        }
        self.width = w;
        self.height = h;
        self.cols = (w + TILE - 1) / TILE;
        self.rows = (h + TILE - 1) / TILE;
        let n = (self.cols * self.rows) as usize;
        self.hashes = vec![None; n];
        self.start = vec![0; n];
        self.occluded = vec![false; n];
        true
    }

    /// Forgets every hash, so the next frame repaints every tile.
    pub fn invalidate(&mut self) {
        self.hashes.iter_mut().for_each(|h| *h = None);
    }

    pub fn len(&self) -> usize {
        (self.cols * self.rows) as usize
    }

    /// Tile `t`'s pixels (clipped to the frame).
    pub fn rect(&self, t: usize) -> IRect {
        tile_rect(t, self.cols, self.width, self.height)
    }

    /// Bins `items` (their tile spans lie inside the grid) in `parts` contiguous parts, on `pool`'s
    /// threads when given: a counting sort, so each tile's list stays in painter's order.
    pub fn bin(&mut self, items: &[Item], parts: usize, pool: Option<&Pool>) {
        let n = self.len();
        let cols = self.cols as usize;
        let parts = parts.clamp(1, items.len().max(1));
        let chunk = items.len().div_ceil(parts).max(1);
        self.counts.clear();
        self.counts.resize(parts * n, 0);
        // Pass 1: each part counts its items per tile.
        let tasks: Vec<_> = self.counts.chunks_mut(n).zip(items.chunks(chunk)).collect();
        for_each(pool, tasks, &|(counts, items): (&mut [u32], &[Item])| {
            for it in items {
                let [x0, y0, x1, y1] = it.tiles.map(|v| v as usize);
                for ty in y0..y1 {
                    for c in &mut counts[ty * cols + x0..ty * cols + x1] {
                        *c += 1;
                    }
                }
            }
        });
        // Each (tile, part) range starts after the earlier parts' entries of the same tile.
        self.offsets.clear();
        self.offsets.resize(n + 1, 0);
        self.cursor.clear();
        self.cursor.resize(parts * n, 0);
        let mut run = 0u32;
        for t in 0..n {
            self.offsets[t] = run;
            for p in 0..parts {
                self.cursor[p * n + t] = run;
                run += self.counts[p * n + t];
            }
        }
        self.offsets[n] = run;
        self.list.clear();
        self.list.resize(run as usize, 0);
        // Pass 2: each part writes its items into its own ranges.
        let list = ListPtr(self.list.as_mut_ptr());
        let tasks: Vec<_> =
            self.cursor.chunks_mut(n).zip(items.chunks(chunk)).enumerate().map(|(p, (c, it))| (p * chunk, c, it)).collect();
        for_each(pool, tasks, &|(base, cursor, items): (usize, &mut [u32], &[Item])| {
            let list = &list;
            for (i, it) in items.iter().enumerate() {
                let [x0, y0, x1, y1] = it.tiles.map(|v| v as usize);
                for ty in y0..y1 {
                    for c in &mut cursor[ty * cols + x0..ty * cols + x1] {
                        // SAFETY: `*c` stays inside this part's range of this tile (pass 1 counted
                        // exactly these entries), and ranges of different (tile, part) pairs are
                        // disjoint, so no other thread writes this slot; the list outlives the job.
                        unsafe { list.0.add(*c as usize).write((base + i) as u32) };
                        *c += 1;
                    }
                }
            }
        });
    }

    /// Cuts each tile's list at its occluder, hashes what remains (with `salt`), and puts the tiles
    /// whose hash changed into `dirty`, ascending. Tiles are split into `parts` ranges for `pool`.
    pub fn diff(&mut self, items: &[Item], salt: u64, dirty: &mut Vec<u32>, parts: usize, pool: Option<&Pool>) {
        let n = self.len();
        let chunk = n.div_ceil(parts.max(1)).max(1);
        let found = Mutex::new(Vec::new());
        let (cols, width, height) = (self.cols, self.width, self.height);
        let (offsets, list) = (&self.offsets, &self.list);
        let tasks: Vec<_> = self
            .hashes
            .chunks_mut(chunk)
            .zip(self.start.chunks_mut(chunk))
            .zip(self.occluded.chunks_mut(chunk))
            .enumerate()
            .map(|(k, ((h, s), o))| (k * chunk, h, s, o))
            .collect();
        for_each(pool, tasks, &|(first, hashes, start, occluded): (usize, &mut [Option<u64>], &mut [u32], &mut [bool])| {
            let mut mine = Vec::new();
            for (j, ((hash, start), occ)) in hashes.iter_mut().zip(start.iter_mut()).zip(occluded.iter_mut()).enumerate() {
                let t = first + j;
                let rect = tile_rect(t, cols, width, height);
                let (b, e) = (offsets[t] as usize, offsets[t + 1] as usize);
                let mut s = b;
                let mut covered = false;
                for k in (b..e).rev() {
                    if items[list[k] as usize].opaque.contains(&rect) {
                        s = k;
                        covered = true;
                        break;
                    }
                }
                let mut h = mix(salt, covered as u64);
                for &i in &list[s..e] {
                    h = mix(h, items[i as usize].sig);
                }
                let h = finish(h);
                *start = s as u32;
                *occ = covered;
                if *hash != Some(h) {
                    *hash = Some(h);
                    mine.push(t as u32);
                }
            }
            if !mine.is_empty() {
                found.lock().unwrap_or_else(|e| e.into_inner()).push(mine);
            }
        });
        let mut found = found.into_inner().unwrap_or_else(|e| e.into_inner());
        found.sort_unstable_by_key(|v| v[0]);
        dirty.clear();
        for v in found {
            dirty.extend_from_slice(&v);
        }
    }

    /// The items tile `t` draws, in order, and whether the first covers it opaquely.
    pub fn entries(&self, t: usize) -> (&[u32], bool) {
        (&self.list[self.start[t] as usize..self.offsets[t + 1] as usize], self.occluded[t])
    }

    /// Merges the dirty tiles (in ascending order) into few rectangles: runs along each tile row,
    /// then runs of equal spans down the rows.
    pub fn damage(&self, dirty: &[u32]) -> Vec<Bounds<DevicePixels>> {
        let mut done: Vec<IRect> = Vec::new();
        let mut open: Vec<IRect> = Vec::new();
        let mut next_open: Vec<IRect> = Vec::new();
        let mut i = 0;
        let mut ty = -1;
        let flush_row = |open: &mut Vec<IRect>, next: &mut Vec<IRect>, done: &mut Vec<IRect>| {
            done.append(open);
            std::mem::swap(open, next);
        };
        while i < dirty.len() {
            let t = dirty[i] as i32;
            let row = t / self.cols;
            if row != ty {
                if row != ty + 1 {
                    done.append(&mut open);
                }
                flush_row(&mut open, &mut next_open, &mut done);
                ty = row;
            }
            let mut j = i;
            while j + 1 < dirty.len() && dirty[j + 1] as i32 == dirty[j] as i32 + 1 && dirty[j + 1] as i32 / self.cols == row {
                j += 1;
            }
            let a = self.rect(t as usize);
            let b = self.rect(dirty[j] as usize);
            let run = IRect::new(a.x0, a.y0, b.x1, a.y1);
            if let Some(k) = open.iter().position(|o| o.x0 == run.x0 && o.x1 == run.x1 && o.y1 == run.y0) {
                let mut o = open.swap_remove(k);
                o.y1 = run.y1;
                next_open.push(o);
            } else {
                next_open.push(run);
            }
            i = j + 1;
        }
        done.append(&mut open);
        done.append(&mut next_open);
        done.into_iter()
            .map(|r| {
                Bounds::new(point(DevicePixels(r.x0), DevicePixels(r.y0)), size(DevicePixels(r.width()), DevicePixels(r.height())))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn damage_merges_runs_and_rows() {
        let mut t = Tiles::default();
        t.resize(64 * 4, 64 * 3 + 10);
        // Tiles (1,0) (2,0) (1,1) (2,1) form one block; (0,3) is a partial bottom tile.
        let d = t.damage(&[1, 2, 5, 6, 12]);
        let r: Vec<_> = d.iter().map(|b| (b.origin.x.0, b.origin.y.0, b.size.width.0, b.size.height.0)).collect();
        assert_eq!(r.len(), 2, "{r:?}");
        assert!(r.contains(&(64, 0, 128, 128)));
        assert!(r.contains(&(0, 192, 64, 10)));
        // Non-adjacent rows do not merge.
        let d = t.damage(&[0, 8]);
        assert_eq!(d.len(), 2);
    }
}
