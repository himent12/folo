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

use crate::{Bounds, DevicePixels, Framebuffer, GlyphBlend, Primitive, Scene};
use geom::IRect;
use items::Item;
use pool::{Pool, for_each};
use prep::{Caches, GlyphTables, Prep, Prepared};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use target::FrameMut;
use tiles::Tiles;

/// Dirty tiles up to this many are drawn on the calling thread.
const INLINE_TILES: usize = 8;
/// Scenes with at least this many primitives are resolved, binned and hashed on the pool.
const PARALLEL_FRONT: usize = 6000;
/// `slot` values of items that `prepare` left to the tiles.
const LAZY: u32 = u32::MAX;
/// The high bit of a `slot` marks a glyph table index.
const GLYPH: u32 = 1 << 31;

/// What the last `render` did, for benches and diagnostics.
#[derive(Clone, Debug, Default)]
pub struct RenderStats {
    /// Primitives in the scene.
    pub items: usize,
    /// Tiles in the frame.
    pub tiles: usize,
    /// Tiles redrawn.
    pub dirty_tiles: usize,
    /// Resolving, binning and hashing.
    pub bin_hash: Duration,
    /// Of `bin_hash`: resolving the primitives.
    pub resolve: Duration,
    /// Of `bin_hash`: binning them into tiles.
    pub bin: Duration,
    /// Preparing the items the dirty tiles draw (cached masks and tables).
    pub prepare: Duration,
    /// Drawing the dirty tiles.
    pub raster: Duration,
    /// Threads that drew (1: the calling thread alone).
    pub threads_used: usize,
}

/// The tile renderer. One per window: it keeps last frame's tile hashes, its caches and its pool.
pub struct Renderer {
    tiles: Tiles,
    /// One per primitive of the scene.
    items: Vec<Item>,
    /// Per item, when `mark[i] == stamp`: an index into `preps`, `GLYPH | table`, or `LAZY`.
    slot: Vec<u32>,
    mark: Vec<u32>,
    stamp: u32,
    preps: Vec<Prep>,
    glyphs: GlyphTables,
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
            glyphs: GlyphTables::default(),
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

    /// Bounds the renderer's caches to about `bytes` in all (default 108 MB: shadows 24, paths 16,
    /// resampled images 64, tables 2 + 2; a frame usually needs a small part of it).
    pub fn set_cache_budget(&mut self, bytes: usize) {
        self.caches.set_budget(bytes);
    }

    /// The next frame repaints everything (after a resize, or when `frame` is a new buffer).
    pub fn invalidate(&mut self) {
        self.tiles.invalidate();
    }

    /// Creates the pool the first time it is needed.
    fn ensure_pool(&mut self) {
        if self.threads > 1 && self.pool.is_none() {
            self.pool = Some(Pool::new(self.threads - 1));
        }
    }

    /// Makes `frame` show `scene`. Returns the rectangles that changed (device px, tile-aligned,
    /// merged). `frame` must be the buffer the previous call drew, or the call must follow `invalidate`.
    pub fn render(
        &mut self,
        scene: &Scene,
        glyph: GlyphBlend,
        frame: &mut Framebuffer,
    ) -> Vec<Bounds<DevicePixels>> {
        let t0 = Instant::now();
        assert_eq!(frame.pixels.len(), frame.width as usize * frame.height as usize, "framebuffer size");
        self.tiles.resize(frame.width, frame.height);
        let frame_rect = IRect::new(0, 0, frame.width as i32, frame.height as i32);
        let prims = &scene.primitives;
        let n = prims.len();

        // Resolve, bin and hash: on the pool for big scenes.
        let parts = if n >= PARALLEL_FRONT { self.threads } else { 1 };
        let mut items = std::mem::take(&mut self.items);
        items.resize(n, Item::default());
        if parts > 1 {
            self.ensure_pool();
        }
        let pool = if parts > 1 { self.pool.as_ref() } else { None };
        let chunk = n.div_ceil(parts).max(1);
        let tasks: Vec<_> = prims.chunks(chunk).zip(items.chunks_mut(chunk)).collect();
        for_each(pool, tasks, &|(p, it): (&[Primitive], &mut [Item])| items::resolve(p, frame_rect, it));
        let ta = Instant::now();
        self.tiles.bin(&items, parts, pool);
        let tb = Instant::now();
        self.tiles.diff(&items, blend_salt(&glyph), &mut self.dirty, parts, pool);
        self.items = items;
        let t1 = Instant::now();

        // Prepare what needs the caches, for the items the dirty tiles draw.
        self.stamp = self.stamp.wrapping_add(1);
        if self.stamp == 0 {
            self.mark.iter_mut().for_each(|m| *m = 0);
            self.stamp = 1;
        }
        if self.mark.len() < n {
            self.mark.resize(n, 0);
            self.slot.resize(n, 0);
        }
        self.preps.clear();
        self.glyphs.clear();
        let mut jobs: Vec<(usize, prep::Job)> = Vec::new();
        let all = self.dirty.len() == self.tiles.len();
        for k in 0..if all { 1 } else { self.dirty.len() } {
            let entries: &[u32] = if all { &[] } else { self.tiles.entries(self.dirty[k] as usize).0 };
            let indices: Box<dyn Iterator<Item = usize>> = if all {
                Box::new((0..n).filter(|&i| !self.items[i].rect.is_empty()))
            } else {
                Box::new(entries.iter().map(|&i| i as usize))
            };
            for i in indices {
                if self.mark[i] == self.stamp {
                    continue;
                }
                self.mark[i] = self.stamp;
                self.slot[i] =
                    match prep::prepare(&prims[i], &mut self.caches, &mut self.glyphs, &glyph, frame_rect) {
                        Prepared::Lazy => LAZY,
                        Prepared::Glyph(g) => GLYPH | g,
                        Prepared::Item(p) => {
                            self.preps.push(p);
                            self.preps.len() as u32 - 1
                        }
                        Prepared::Pending(job) => {
                            jobs.push((i, job));
                            LAZY
                        }
                    };
            }
        }
        // Cache misses (shadow masks, resampled images, path coverage) are made on the pool.
        if !jobs.is_empty() {
            let mut made: Vec<Option<prep::Made>> = (0..jobs.len()).map(|_| None).collect();
            let parallel = jobs.len() > 1 && self.threads > 1;
            if parallel {
                self.ensure_pool();
            }
            let tasks: Vec<_> = made.iter_mut().zip(&jobs).collect();
            for_each(if parallel { self.pool.as_ref() } else { None }, tasks, &|(out, (_, job))| {
                *out = Some(job.run());
            });
            for ((i, job), m) in jobs.drain(..).zip(made) {
                let p = job.finish(m.expect("every job ran"), &prims[i], &mut self.caches, frame_rect);
                self.preps.push(p);
                self.slot[i] = self.preps.len() as u32 - 1;
            }
        }
        let t2 = Instant::now();

        // Draw the dirty tiles.
        let fm = FrameMut::new(&mut frame.pixels, frame.width);
        let inline = self.dirty.len() <= INLINE_TILES || self.threads == 1;
        if !inline {
            self.ensure_pool();
        }
        let pool = if inline { None } else { self.pool.as_ref() };
        let threads_used = pool.map_or(1, Pool::threads);
        let ctx = TileCtx {
            tiles: &self.tiles,
            items: &self.items,
            slot: &self.slot,
            preps: &self.preps,
            glyphs: &self.glyphs.tables,
            prims,
        };
        match pool {
            None => {
                for &t in &self.dirty {
                    ctx.draw_tile(t as usize, &fm);
                }
            }
            Some(pool) => {
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
        }
        let t3 = Instant::now();

        self.caches.end_frame();
        self.stats = RenderStats {
            items: n,
            tiles: self.tiles.len(),
            dirty_tiles: self.dirty.len(),
            bin_hash: t1 - t0,
            resolve: ta - t0,
            bin: tb - ta,
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
    glyphs: &'a [(std::sync::Arc<sprite::AlphaLut>, blend::Tint)],
    prims: &'a [Primitive],
}

impl TileCtx<'_> {
    /// Draws tile `t`. The caller hands each dirty tile to exactly one thread.
    fn draw_tile(&self, t: usize, fm: &FrameMut) {
        let rect = self.tiles.rect(t);
        // SAFETY: `rect` is a tile of this frame (inside it), and each tile goes to one thread only,
        // so no other target overlaps it.
        let mut target = unsafe { fm.target(rect) };
        let (entries, occluded) = self.tiles.entries(t);
        if !occluded {
            // GPUI clears its render target to transparent black; the window is opaque.
            target.fill(0);
        }
        // Later opaque quads hide what is under them: each item's rectangle is cut by the opaque
        // cores after it that span it (only pixels that would be overwritten are skipped).
        let mut covers: [(usize, IRect); 8] = [(0, IRect::EMPTY); 8];
        let mut nc = 0;
        for (k, &i) in entries.iter().enumerate().skip(1) {
            let c = self.items[i as usize].opaque.intersect(&rect);
            if !c.is_empty() && nc < covers.len() {
                covers[nc] = (k, c);
                nc += 1;
            }
        }
        for (k, &i) in entries.iter().enumerate() {
            let i = i as usize;
            let mut r = self.items[i].rect.intersect(&rect);
            for (ck, c) in &covers[..nc] {
                if *ck > k {
                    r = cut(r, c);
                }
            }
            if r.is_empty() {
                continue;
            }
            let slot = self.slot[i];
            let (prep, glyph) = match slot {
                LAZY => (None, 0),
                s if s & GLYPH != 0 => (None, s & !GLYPH),
                s => (Some(&self.preps[s as usize]), 0),
            };
            prep::draw(&mut target, &self.prims[i], prep, glyph, self.glyphs, r);
        }
    }
}

/// `r` without the rows or columns of it that `c` covers from one side to the other.
fn cut(mut r: IRect, c: &IRect) -> IRect {
    if c.x0 <= r.x0 && c.x1 >= r.x1 {
        if c.y0 <= r.y0 {
            r.y0 = r.y0.max(c.y1);
        }
        if c.y1 >= r.y1 {
            r.y1 = r.y1.min(c.y0);
        }
    }
    if c.y0 <= r.y0 && c.y1 >= r.y1 {
        if c.x0 <= r.x0 {
            r.x0 = r.x0.max(c.x1);
        }
        if c.x1 >= r.x1 {
            r.x1 = r.x1.min(c.x0);
        }
    }
    r
}

/// What the glyph blend contributes to every tile hash (a new blend repaints everything).
fn blend_salt(b: &GlyphBlend) -> u64 {
    let mut h = hash::mix(0x7a11, b.apply as u64);
    h = hash::mix_f2(
        hash::mix_f2(h, b.gamma_ratios[0], b.gamma_ratios[1]),
        b.gamma_ratios[2],
        b.gamma_ratios[3],
    );
    hash::finish(hash::mix(h, b.grayscale_enhanced_contrast.to_bits() as u64))
}
