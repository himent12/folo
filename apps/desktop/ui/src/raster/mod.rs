//! The CPU tile renderer (`ui/DESIGN.md`, Renderer contract): a `Scene` becomes pixels in a
//! `Framebuffer`, touching only the 64x64 tiles whose content changed.
//!
//! A frame: resolve every primitive once (pixel rectangle, occlusion rectangle, signature), bin them
//! into tiles, cut each tile's list at its topmost opaque cover, hash it, and compare with last
//! frame's hash. Only the items the dirty tiles draw are prepared (tables, cached masks), and only the
//! dirty tiles are drawn: a few on the calling thread, more on the renderer's own pool. Each pixel's
//! value depends only on the primitives over it, never on how the frame was split, so incremental
//! frames equal fresh ones byte for byte, for every thread count.

mod blend;
mod cache;
pub mod debug;
mod geom;
mod gradient;
mod hash;
mod image;
mod items;
mod path;
mod pool;
mod prep;
mod quad;
mod quad_eval;
#[cfg(test)]
mod quad_tests;
mod shadow;
mod sprite;
mod target;
mod tiles;

pub use cache::{Mask, MaskCache};
pub use tiles::TILE as TILE_SIZE;

use crate::{Bounds, DevicePixels, Framebuffer, GlyphBlend, Scene};
use geom::IRect;
use items::Item;
use pool::Pool;
use prep::{Caches, LutMemo, Prep};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use target::FrameMut;
use tiles::Tiles;

/// Dirty tiles up to this many are drawn on the calling thread.
const INLINE_TILES: usize = 8;

/// What the last `render` did, for benches and diagnostics.
#[derive(Clone, Debug, Default)]
pub struct RenderStats {
    /// Visible primitives.
    pub items: usize,
    /// Tiles in the frame.
    pub tiles: usize,
    /// Tiles redrawn.
    pub dirty_tiles: usize,
    /// Resolving, binning and hashing.
    pub bin_hash: Duration,
    /// Preparing the items the dirty tiles draw (tables, masks, caches).
    pub prepare: Duration,
    /// Drawing the dirty tiles.
    pub raster: Duration,
    /// Threads that drew (1: the calling thread alone).
    pub threads_used: usize,
}

/// The tile renderer. One per window: it keeps last frame's tile hashes, its caches and its pool.
pub struct Renderer {
    tiles: Tiles,
    items: Vec<Item>,
    /// `slot[i]` indexes `preps` for item `i` when `mark[i] == stamp`.
    slot: Vec<u32>,
    mark: Vec<u32>,
    stamp: u32,
    preps: Vec<Prep>,
    dirty: Vec<u32>,
    caches: Caches,
    pool: Option<Pool>,
    threads: usize,
    stats: RenderStats,
}

impl Default for Renderer {
    fn default() -> Self {
        Self::new()
    }
}

impl Renderer {
    /// A renderer drawing with up to `min(8, available_parallelism)` threads.
    pub fn new() -> Self {
        let n = std::thread::available_parallelism().map_or(1, |n| n.get());
        Self::with_threads(n.min(8))
    }

    /// A renderer drawing with up to `threads` threads (1: always on the calling thread).
    pub fn with_threads(threads: usize) -> Self {
        Renderer {
            tiles: Tiles::default(),
            items: Vec::new(),
            slot: Vec::new(),
            mark: Vec::new(),
            stamp: 0,
            preps: Vec::new(),
            dirty: Vec::new(),
            caches: Caches::new(),
            pool: None,
            threads: threads.max(1),
            stats: RenderStats::default(),
        }
    }

    /// The most threads a frame uses.
    pub fn threads(&self) -> usize {
        self.threads
    }

    /// What the last frame did.
    pub fn stats(&self) -> &RenderStats {
        &self.stats
    }

    /// Bytes held by the renderer's caches (shadow, path and image copies, glyph and gradient tables).
    pub fn cache_bytes(&self) -> usize {
        self.caches.bytes()
    }

    /// The next frame repaints everything (after a resize, or when `frame` is a new buffer).
    pub fn invalidate(&mut self) {
        self.tiles.invalidate();
    }

    /// Makes `frame` show `scene`. Returns the rectangles that changed (device px, tile-aligned,
    /// merged). `frame` must be the buffer the previous call drew, or the call must follow `invalidate`.
    pub fn render(&mut self, scene: &Scene, glyph: GlyphBlend, frame: &mut Framebuffer) -> Vec<Bounds<DevicePixels>> {
        let t0 = Instant::now();
        assert_eq!(frame.pixels.len(), frame.width as usize * frame.height as usize, "framebuffer size");
        self.tiles.resize(frame.width, frame.height);
        let frame_rect = IRect::new(0, 0, frame.width as i32, frame.height as i32);
        let prims = &scene.primitives;
        items::resolve(prims, frame_rect, &mut self.items);
        self.tiles.bin(&self.items);
        self.tiles.diff(&self.items, blend_salt(&glyph), &mut self.dirty);
        let t1 = Instant::now();

        // Prepare what the dirty tiles draw.
        self.stamp = self.stamp.wrapping_add(1);
        if self.stamp == 0 {
            self.mark.iter_mut().for_each(|m| *m = 0);
            self.stamp = 1;
        }
        if self.mark.len() < self.items.len() {
            self.mark.resize(self.items.len(), 0);
            self.slot.resize(self.items.len(), 0);
        }
        self.preps.clear();
        let mut memo = LutMemo::new();
        for &t in &self.dirty {
            let (entries, _) = self.tiles.entries(t as usize);
            for &i in entries {
                let i = i as usize;
                if self.mark[i] != self.stamp {
                    self.mark[i] = self.stamp;
                    self.slot[i] = self.preps.len() as u32;
                    let p = prep::prepare(&self.items[i], prims, &mut self.caches, &mut memo, &glyph, frame_rect);
                    self.preps.push(p);
                }
            }
        }
        let t2 = Instant::now();

        // Draw the dirty tiles.
        let fm = FrameMut::new(&mut frame.pixels, frame.width);
        let ctx = TileCtx { tiles: &self.tiles, items: &self.items, slot: &self.slot, preps: &self.preps, scene };
        let mut threads_used = 1;
        if self.dirty.len() <= INLINE_TILES || self.threads == 1 {
            for &t in &self.dirty {
                ctx.draw_tile(t as usize, &fm);
            }
        } else {
            let pool = self.pool.get_or_insert_with(|| Pool::new(self.threads - 1));
            threads_used = pool.threads();
            let next = AtomicUsize::new(0);
            let dirty = &self.dirty;
            pool.run(&|| {
                loop {
                    let k = next.fetch_add(1, Ordering::Relaxed);
                    let Some(&t) = dirty.get(k) else { break };
                    ctx.draw_tile(t as usize, &fm);
                }
            });
        }
        let t3 = Instant::now();

        self.caches.end_frame();
        self.stats = RenderStats {
            items: self.items.len(),
            tiles: self.tiles.len(),
            dirty_tiles: self.dirty.len(),
            bin_hash: t1 - t0,
            prepare: t2 - t1,
            raster: t3 - t2,
            threads_used,
        };
        self.tiles.damage(&self.dirty)
    }
}

/// Everything a tile reads while it draws (shared by the threads).
struct TileCtx<'a> {
    tiles: &'a Tiles,
    items: &'a [Item],
    slot: &'a [u32],
    preps: &'a [Prep],
    scene: &'a Scene,
}

impl TileCtx<'_> {
    /// Draws tile `t`. Each dirty tile is handed to exactly one thread.
    fn draw_tile(&self, t: usize, fm: &FrameMut) {
        let rect = self.tiles.rect(t);
        // SAFETY: `rect` is a tile of this frame (inside it), and the caller hands each tile to one
        // thread only, so no other target overlaps it.
        let mut target = unsafe { fm.target(rect) };
        let (entries, occluded) = self.tiles.entries(t);
        if !occluded {
            // GPUI clears its render target to transparent black; the window is opaque.
            target.fill(0);
        }
        for &i in entries {
            let item = &self.items[i as usize];
            let r = item.rect.intersect(&rect);
            let prep = &self.preps[self.slot[i as usize] as usize];
            prep::draw(&mut target, prep, &self.scene.primitives[item.prim as usize], r);
        }
    }
}

/// What the glyph blend contributes to every tile hash (a new blend repaints everything).
fn blend_salt(b: &GlyphBlend) -> u64 {
    let mut h = hash::mix(0x7a11, b.apply as u64);
    h = hash::mix_f2(hash::mix_f2(h, b.gamma_ratios[0], b.gamma_ratios[1]), b.gamma_ratios[2], b.gamma_ratios[3]);
    hash::finish(hash::mix(h, b.grayscale_enhanced_contrast.to_bits() as u64))
}
