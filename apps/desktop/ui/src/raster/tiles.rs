//! Tiles: 64x64 device pixels. Items are binned by their pixel rectangle, each tile's ordered list is
//! cut at the topmost item that covers it with an opaque colour, the rest is hashed, and the tiles
//! whose hash changed are the frame's damage.

use super::geom::IRect;
use super::hash::{finish, mix};
use super::items::Item;
use crate::{Bounds, DevicePixels, point, size};

/// The tile edge in device pixels.
pub const TILE: i32 = 64;

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
        let (tx, ty) = (t as i32 % self.cols, t as i32 / self.cols);
        IRect::new(tx * TILE, ty * TILE, ((tx + 1) * TILE).min(self.width), ((ty + 1) * TILE).min(self.height))
    }

    fn tile_span(&self, r: &IRect) -> (i32, i32, i32, i32) {
        (r.x0 / TILE, r.y0 / TILE, (r.x1 + TILE - 1) / TILE, (r.y1 + TILE - 1) / TILE)
    }

    /// Bins `items` (their rectangles lie inside the frame).
    pub fn bin(&mut self, items: &[Item]) {
        let n = self.len();
        self.offsets.clear();
        self.offsets.resize(n + 1, 0);
        for it in items {
            let (x0, y0, x1, y1) = self.tile_span(&it.rect);
            for ty in y0..y1 {
                let row = (ty * self.cols) as usize;
                for tx in x0..x1 {
                    self.offsets[row + tx as usize + 1] += 1;
                }
            }
        }
        for t in 0..n {
            self.offsets[t + 1] += self.offsets[t];
        }
        self.cursor.clear();
        self.cursor.extend_from_slice(&self.offsets[..n]);
        self.list.clear();
        self.list.resize(self.offsets[n] as usize, 0);
        for (i, it) in items.iter().enumerate() {
            let (x0, y0, x1, y1) = self.tile_span(&it.rect);
            for ty in y0..y1 {
                let row = (ty * self.cols) as usize;
                for tx in x0..x1 {
                    let c = &mut self.cursor[row + tx as usize];
                    self.list[*c as usize] = i as u32;
                    *c += 1;
                }
            }
        }
    }

    /// Cuts each tile's list at its occluder, hashes what remains (with `salt`), and appends the tiles
    /// whose hash changed to `dirty`.
    pub fn diff(&mut self, items: &[Item], salt: u64, dirty: &mut Vec<u32>) {
        dirty.clear();
        for t in 0..self.len() {
            let rect = self.rect(t);
            let (b, e) = (self.offsets[t] as usize, self.offsets[t + 1] as usize);
            let mut start = b;
            let mut occluded = false;
            for k in (b..e).rev() {
                if items[self.list[k] as usize].opaque.contains(&rect) {
                    start = k;
                    occluded = true;
                    break;
                }
            }
            let mut h = mix(salt, occluded as u64);
            for &i in &self.list[start..e] {
                h = mix(h, items[i as usize].sig);
            }
            let h = finish(h);
            self.start[t] = start as u32;
            self.occluded[t] = occluded;
            if self.hashes[t] != Some(h) {
                self.hashes[t] = Some(h);
                dirty.push(t as u32);
            }
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
