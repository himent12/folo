//! `--bench-frames <n>`: the frame bench. The app runs headless (as `--shot` does, with `--width`,
//! `--height`, `--scale`, `--text`), settles the way `--shot` settles, then draws `n` full redraws (the window
//! marked dirty, nothing changed) and `n` animation ticks (the fake clock moved by one 240 Hz refresh, the
//! window asked for its next frame), and prints one line per kind with the median of every phase
//! (`FrameTimes`), the median allocations per frame, and the frame's primitive, element and hitbox counts
//! and the lines shaped. Owner tag: `"runtime"`.
//!
//! Allocations are counted only when the binary installs [`CountingAlloc`] as its global allocator (the app
//! does with its `bench-alloc` feature); otherwise they print as `-`.

use std::{
    alloc::{GlobalAlloc, Layout, System},
    rc::Rc,
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
    time::{Duration, Instant},
};

use anyhow::{Context as _, Result, anyhow};

use crate::{
    AppCell, AppContext, FrameTimes, HeadlessApp, HeadlessOptions, RequestFrameOptions, TextBackend,
    platform::headless::HeadlessPlatform, px, size,
};

/// One refresh of the 240 Hz display the app is measured on.
const TICK: Duration = Duration::from_nanos(1_000_000_000 / 240);

/// A global allocator that counts allocations (and reallocations) on top of the system allocator, for
/// `--bench-frames`. Install it with `#[global_allocator]` in the binary that runs the bench.
pub struct CountingAlloc;

static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
/// Bytes allocated and not freed (the live heap).
static LIVE_BYTES: AtomicU64 = AtomicU64::new(0);
static COUNTING: AtomicBool = AtomicBool::new(false);

// SAFETY: every method forwards to `System` with the caller's arguments unchanged; the counter is a
// relaxed atomic and allocates nothing.
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        COUNTING.store(true, Ordering::Relaxed);
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        LIVE_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        // SAFETY: the caller upholds `GlobalAlloc::alloc`'s contract, which `System` shares.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        LIVE_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        // SAFETY: as `alloc`.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE_BYTES.fetch_sub(layout.size() as u64, Ordering::Relaxed);
        // SAFETY: `ptr` came from this allocator, so from `System`, with `layout`.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        LIVE_BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
        LIVE_BYTES.fetch_sub(layout.size() as u64, Ordering::Relaxed);
        // SAFETY: as `dealloc`, and the caller upholds `realloc`'s size contract.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

/// The live heap, when a `CountingAlloc` is installed.
pub(crate) fn live_bytes() -> Option<u64> {
    COUNTING.load(Ordering::Relaxed).then(|| LIVE_BYTES.load(Ordering::Relaxed))
}

/// Allocations so far, when a `CountingAlloc` is installed.
fn allocations() -> Option<u64> {
    COUNTING.load(Ordering::Relaxed).then(|| ALLOCATIONS.load(Ordering::Relaxed))
}

/// A `--bench-frames` request from the command line.
pub(crate) struct BenchArgs {
    /// Frames of each kind.
    pub(crate) frames: usize,
    /// The headless window's size, scale and text backend.
    pub(crate) options: HeadlessOptions,
}

impl BenchArgs {
    /// The request in `args` (program name first), if `--bench-frames` is there.
    pub(crate) fn parse(args: &[String]) -> Result<Option<Self>> {
        let value = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1));
        let Some(frames) = value("--bench-frames") else { return Ok(None) };
        let frames: usize = frames.parse().with_context(|| format!("--bench-frames {frames}"))?;
        anyhow::ensure!(frames > 0, "--bench-frames must be positive");
        let number = |name: &str, default: f32| -> Result<f32> {
            value(name).map_or(Ok(default), |v| v.parse().with_context(|| format!("{name} {v}")))
        };
        let text = match value("--text").map(String::as_str) {
            None | Some("native") => TextBackend::Native,
            Some("portable") => TextBackend::Portable,
            Some(other) => return Err(anyhow!("--text {other}: expected `native` or `portable`")),
        };
        let scale_factor = number("--scale", 1.0)?;
        let (width, height) = (number("--width", 1280.0)?, number("--height", 800.0)?);
        super::shot::check_size(width, height, scale_factor)?;
        Ok(Some(BenchArgs {
            frames,
            options: HeadlessOptions { size: size(px(width), px(height)), scale_factor, text },
        }))
    }
}

/// A bench run: the request and the headless platform it runs on.
pub(crate) struct Bench {
    /// Frames of each kind.
    pub(crate) frames: usize,
    /// The window's size and scale (printed).
    pub(crate) options: HeadlessOptions,
    /// The platform.
    pub(crate) platform: Rc<HeadlessPlatform>,
    /// The live heap before and after the app's launch callback (with `CountingAlloc`).
    pub(crate) heap: [Option<u64>; 2],
}

/// One measured frame.
struct Sample {
    /// The phases (all zero when nothing was drawn).
    times: FrameTimes,
    /// The whole `on_request_frame` call, effects included.
    wall_us: u32,
    /// Allocations during it.
    allocations: Option<u64>,
}

impl Bench {
    /// Settles the app's first window as `--shot` does, then measures and prints.
    pub(crate) fn run(self, app: Rc<AppCell>) -> Result<()> {
        let mut harness = HeadlessApp::from_parts(app, self.platform);
        harness.run_until_idle();
        let window = *harness.windows().first().context("the app opened no window")?;
        harness.settle_advancing(window, super::shot::SETTLE_TIME, super::shot::SETTLE_STEP);
        let shaped_before = shaped_lines(&harness, window);

        let measure = |harness: &mut HeadlessApp, dirty: bool| -> Sample {
            if dirty {
                harness.app.update(|cx| cx.update_window(window, |_, window, _| window.refresh())).ok();
            } else {
                harness.advance_clock(TICK);
            }
            let allocations = allocations();
            let start = Instant::now();
            let times = harness
                .app
                .update(|cx| {
                    cx.update_window(window, |_, window, cx| {
                        let frame = window.frame_times.frame;
                        window.on_request_frame(RequestFrameOptions::default(), cx);
                        let times = window.frame_times;
                        if times.frame == frame {
                            FrameTimes { frame, ..FrameTimes::default() }
                        } else {
                            times
                        }
                    })
                })
                .unwrap_or_default();
            let wall_us = start.elapsed().as_micros() as u32;
            let allocations = allocations.zip(self::allocations()).map(|(a, b)| b - a);
            Sample { times, wall_us, allocations }
        };
        // Warm up: caches, pools, buffers at their steady size.
        for _ in 0..3 {
            measure(&mut harness, true);
        }
        let full: Vec<Sample> = (0..self.frames).map(|_| measure(&mut harness, true)).collect();
        let shaped_full = shaped_lines(&harness, window) - shaped_before;
        let ticks: Vec<Sample> = (0..self.frames).map(|_| measure(&mut harness, false)).collect();
        let shaped_ticks = shaped_lines(&harness, window) - shaped_before - shaped_full;

        let last = full.last().map(|s| s.times).unwrap_or_default();
        let (w, h, s) = (self.options.size.width.0, self.options.size.height.0, self.options.scale_factor);
        println!(
            "bench-frames {w}x{h}@{s}: {} prims, {} elements, {} hitboxes, {} frames each",
            last.primitives, last.elements, last.hitboxes, self.frames
        );
        print_line("full", &full, shaped_full);
        print_line("tick", &ticks, shaped_ticks);
        print_memory(&harness, window, self.heap);
        Ok(())
    }
}

/// Prints where the heap is: the whole live heap (with `CountingAlloc`) and the window's big holders.
fn print_memory(harness: &HeadlessApp, window: crate::AnyWindowHandle, heap: [Option<u64>; 2]) {
    let mb = |bytes: usize| bytes as f64 / (1 << 20) as f64;
    let app = harness.app.borrow();
    let Some(Some(window)) = app.windows.get(&window.id) else { return };
    let mb_of = |bytes: Option<u64>| bytes.map_or("-".to_string(), |b| format!("{:.1}", mb(b as usize)));
    let (start, launched, now) = (mb_of(heap[0]), mb_of(heap[1]), mb_of(live_bytes()));
    let framebuffer = window.framebuffer.pixels.capacity() * 4;
    let layout = window.layout_engine.as_ref().map_or(0, |engine| engine.tree.heap_bytes());
    println!(
        "  memory: heap {now} MB (at start {start}, after the launch callback {launched}) | framebuffer {:.1} MB, renderer caches {:.1} MB, glyph masks {:.1} MB ({}), icon masks {:.1} MB ({}), layout tree {:.1} MB, scenes {:.1} MB",
        mb(framebuffer),
        mb(window.renderer.cache_bytes()),
        mb(window.glyph_masks.bytes()),
        window.glyph_masks.len(),
        mb(window.icon_masks.bytes()),
        window.icon_masks.len(),
        mb(layout),
        mb((window.rendered_frame.scene.primitives.capacity()
            + window.next_frame.scene.primitives.capacity())
            * std::mem::size_of::<crate::Primitive>()),
    );
}

/// Lines the window's text system has shaped so far.
fn shaped_lines(harness: &HeadlessApp, window: crate::AnyWindowHandle) -> usize {
    match harness.app.borrow().windows.get(&window.id) {
        Some(Some(window)) => window.text_system.shaped_line_count(),
        _ => 0,
    }
}

/// Prints the medians of `samples` as one line.
fn print_line(kind: &str, samples: &[Sample], shaped: usize) {
    let median = |f: &dyn Fn(&Sample) -> u64| -> u64 {
        let mut v: Vec<u64> = samples.iter().map(f).collect();
        v.sort_unstable();
        v.get(v.len() / 2).copied().unwrap_or_default()
    };
    let drawn = samples.iter().filter(|s| s.times.frame != 0 && s.times.total_us() != 0).count();
    let allocations = if samples.iter().all(|s| s.allocations.is_some()) {
        median(&|s| s.allocations.unwrap_or_default()).to_string()
    } else {
        "-".to_string()
    };
    println!(
        "  {kind}: render {} build {} layout {} prepaint {} paint {} finish {} raster {} present {} total {} (min {}) wall {} us | allocs {} | drawn {}/{} | shaped {}",
        median(&|s| s.times.render_us.into()),
        median(&|s| s.times.build_us.into()),
        median(&|s| s.times.layout_us.into()),
        median(&|s| s.times.prepaint_us.into()),
        median(&|s| s.times.paint_us.into()),
        median(&|s| s.times.finish_us.into()),
        median(&|s| s.times.raster_us.into()),
        median(&|s| s.times.present_us.into()),
        median(&|s| s.times.total_us().into()),
        samples.iter().map(|s| s.times.total_us()).filter(|&t| t > 0).min().unwrap_or_default(),
        median(&|s| s.wall_us.into()),
        allocations,
        drawn,
        samples.len(),
        shaped,
    );
}
