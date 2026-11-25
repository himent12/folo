// Adapted from GPUI (Apache-2.0): crates/gpui/src/text_system/line_layout.rs (LineLayoutCache)
//! The per-window line layout cache: layouts used in the previous frame are kept and reused, and
//! `finish_frame` drops anything the frame did not use, so a frame whose text did not change shapes
//! nothing and does no per-hit map removal/re-insertion.
//!
//! A key holds everything a layout depends on: the text, size, font runs, wrap width and line limit (a
//! wrapped layout's boundaries stop at the limit; GPUI's key leaves it out, so the first clamp asked
//! for a text and width won, even in later frames), and the text system's font generation (added fonts
//! can change which face a character falls back to: `TextSystem`'s module docs).

use super::fx_hash::FxHashMap;
use crate::{FontRun, LineLayout, Pixels, PlatformTextSystem, SharedString, WrappedLineLayout, px};
use parking_lot::Mutex;
use smallvec::SmallVec;
use std::borrow::Borrow;
use std::hash::{Hash, Hasher};
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

pub(crate) struct LineLayoutCache {
    state: Mutex<CacheState>,
    platform_text_system: Arc<dyn PlatformTextSystem>,
    /// How many lines the platform has shaped (tests check that an unchanged frame shapes none).
    shaped_lines: AtomicUsize,
}

struct CacheEntry<T> {
    key: Arc<CacheKey>,
    layout: Arc<T>,
    last_used_frame: u64,
}

#[derive(Default)]
struct CacheState {
    frame: u64,
    lines: FxHashMap<Arc<CacheKey>, CacheEntry<LineLayout>>,
    wrapped_lines: FxHashMap<Arc<CacheKey>, CacheEntry<WrappedLineLayout>>,
    previous_used_lines: Vec<Arc<CacheKey>>,
    previous_used_wrapped_lines: Vec<Arc<CacheKey>>,
    current_used_lines: Vec<Arc<CacheKey>>,
    current_used_wrapped_lines: Vec<Arc<CacheKey>>,
}

/// A position in the current frame's list of used layouts (for reusing a cached element's layouts).
#[derive(Clone, Default, Debug, PartialEq, Eq)]
pub struct LineLayoutIndex {
    lines_index: usize,
    wrapped_lines_index: usize,
}

impl LineLayoutCache {
    pub(crate) fn new(platform_text_system: Arc<dyn PlatformTextSystem>) -> Self {
        Self { state: Mutex::default(), platform_text_system, shaped_lines: AtomicUsize::new(0) }
    }

    pub(crate) fn shaped_lines(&self) -> usize {
        self.shaped_lines.load(Ordering::Relaxed)
    }

    pub(crate) fn layout_index(&self) -> LineLayoutIndex {
        let state = self.state.lock();
        LineLayoutIndex {
            lines_index: state.current_used_lines.len(),
            wrapped_lines_index: state.current_used_wrapped_lines.len(),
        }
    }

    pub(crate) fn reuse_layouts(&self, range: Range<LineLayoutIndex>) {
        let mut guard = self.state.lock();
        let state = &mut *guard;
        let frame = state.frame;

        for key in &state.previous_used_lines[range.start.lines_index..range.end.lines_index] {
            if let Some(entry) = state.lines.get_mut(key) {
                entry.last_used_frame = frame;
            }
            state.current_used_lines.push(key.clone());
        }

        for key in
            &state.previous_used_wrapped_lines[range.start.wrapped_lines_index..range.end.wrapped_lines_index]
        {
            if let Some(entry) = state.wrapped_lines.get_mut(key) {
                entry.last_used_frame = frame;
            }
            state.current_used_wrapped_lines.push(key.clone());
        }
    }

    pub(crate) fn truncate_layouts(&self, index: LineLayoutIndex) {
        let mut state = self.state.lock();
        state.current_used_lines.truncate(index.lines_index);
        state.current_used_wrapped_lines.truncate(index.wrapped_lines_index);
    }

    pub(crate) fn finish_frame(&self) {
        let mut guard = self.state.lock();
        let state = &mut *guard;
        let frame = state.frame;
        state.lines.retain(|_, entry| entry.last_used_frame == frame);
        state.wrapped_lines.retain(|_, entry| entry.last_used_frame == frame);
        std::mem::swap(&mut state.previous_used_lines, &mut state.current_used_lines);
        state.current_used_lines.clear();
        std::mem::swap(&mut state.previous_used_wrapped_lines, &mut state.current_used_wrapped_lines);
        state.current_used_wrapped_lines.clear();
        state.frame = frame.wrapping_add(1);
    }

    pub(crate) fn layout_wrapped_line(
        &self,
        text: &str,
        into_shared: impl FnOnce() -> SharedString,
        font_size: Pixels,
        runs: &[FontRun],
        wrap_width: Option<Pixels>,
        max_lines: Option<usize>,
        font_generation: u64,
    ) -> Arc<WrappedLineLayout> {
        // Without a wrap width there are no boundaries to limit.
        let max_lines = wrap_width.and(max_lines);
        let key =
            &CacheKeyRef { text, font_size, runs, wrap_width, max_lines, force_width: None, font_generation }
                as &dyn AsCacheKeyRef;

        {
            let mut guard = self.state.lock();
            let state = &mut *guard;
            if let Some(entry) = state.wrapped_lines.get_mut(key) {
                if entry.last_used_frame != state.frame {
                    entry.last_used_frame = state.frame;
                    state.current_used_wrapped_lines.push(entry.key.clone());
                }
                return entry.layout.clone();
            }
        }

        let text = into_shared();
        let unwrapped_layout =
            self.layout_line(&text, || text.clone(), font_size, runs, None, font_generation);
        let wrap_boundaries = if let Some(wrap_width) = wrap_width {
            unwrapped_layout.compute_wrap_boundaries(text.as_ref(), wrap_width, max_lines)
        } else {
            SmallVec::new()
        };
        let layout = Arc::new(WrappedLineLayout { unwrapped_layout, wrap_boundaries, wrap_width });
        let key = Arc::new(CacheKey {
            text,
            font_size,
            runs: SmallVec::from(runs),
            wrap_width,
            max_lines,
            force_width: None,
            font_generation,
        });

        let mut guard = self.state.lock();
        let state = &mut *guard;
        if let Some(entry) = state.wrapped_lines.get_mut(key.as_ref() as &dyn AsCacheKeyRef) {
            if entry.last_used_frame != state.frame {
                entry.last_used_frame = state.frame;
                state.current_used_wrapped_lines.push(entry.key.clone());
            }
            return entry.layout.clone();
        }
        state.wrapped_lines.insert(
            key.clone(),
            CacheEntry { key: key.clone(), layout: layout.clone(), last_used_frame: state.frame },
        );
        state.current_used_wrapped_lines.push(key);
        layout
    }

    pub(crate) fn layout_line(
        &self,
        text: &str,
        into_shared: impl FnOnce() -> SharedString,
        font_size: Pixels,
        runs: &[FontRun],
        force_width: Option<Pixels>,
        font_generation: u64,
    ) -> Arc<LineLayout> {
        let key = &CacheKeyRef {
            text,
            font_size,
            runs,
            wrap_width: None,
            max_lines: None,
            force_width,
            font_generation,
        } as &dyn AsCacheKeyRef;

        {
            let mut guard = self.state.lock();
            let state = &mut *guard;
            if let Some(entry) = state.lines.get_mut(key) {
                if entry.last_used_frame != state.frame {
                    entry.last_used_frame = state.frame;
                    state.current_used_lines.push(entry.key.clone());
                }
                return entry.layout.clone();
            }
        }

        let text = into_shared();
        self.shaped_lines.fetch_add(1, Ordering::Relaxed);
        let mut layout = self.platform_text_system.layout_line(&text, font_size, runs);

        if let Some(force_width) = force_width {
            let mut glyph_pos = 0;
            for run in layout.runs.iter_mut() {
                for glyph in run.glyphs.iter_mut() {
                    if (glyph.position.x - glyph_pos * force_width).abs() > px(1.) {
                        glyph.position.x = glyph_pos * force_width;
                    }
                    glyph_pos += 1;
                }
            }
        }

        let key = Arc::new(CacheKey {
            text,
            font_size,
            runs: SmallVec::from(runs),
            wrap_width: None,
            max_lines: None,
            force_width,
            font_generation,
        });
        let layout = Arc::new(layout);

        let mut guard = self.state.lock();
        let state = &mut *guard;
        if let Some(entry) = state.lines.get_mut(key.as_ref() as &dyn AsCacheKeyRef) {
            if entry.last_used_frame != state.frame {
                entry.last_used_frame = state.frame;
                state.current_used_lines.push(entry.key.clone());
            }
            return entry.layout.clone();
        }
        state.lines.insert(
            key.clone(),
            CacheEntry { key: key.clone(), layout: layout.clone(), last_used_frame: state.frame },
        );
        state.current_used_lines.push(key);
        layout
    }
}

trait AsCacheKeyRef {
    fn as_cache_key_ref(&self) -> CacheKeyRef<'_>;
}

#[derive(Clone, Debug, Eq)]
struct CacheKey {
    text: SharedString,
    font_size: Pixels,
    runs: SmallVec<[FontRun; 1]>,
    wrap_width: Option<Pixels>,
    max_lines: Option<usize>,
    force_width: Option<Pixels>,
    font_generation: u64,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
struct CacheKeyRef<'a> {
    text: &'a str,
    font_size: Pixels,
    runs: &'a [FontRun],
    wrap_width: Option<Pixels>,
    max_lines: Option<usize>,
    force_width: Option<Pixels>,
    font_generation: u64,
}

impl PartialEq for dyn AsCacheKeyRef + '_ {
    fn eq(&self, other: &dyn AsCacheKeyRef) -> bool {
        self.as_cache_key_ref() == other.as_cache_key_ref()
    }
}

impl Eq for dyn AsCacheKeyRef + '_ {}

impl Hash for dyn AsCacheKeyRef + '_ {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_cache_key_ref().hash(state)
    }
}

impl AsCacheKeyRef for CacheKey {
    fn as_cache_key_ref(&self) -> CacheKeyRef<'_> {
        CacheKeyRef {
            text: &self.text,
            font_size: self.font_size,
            runs: self.runs.as_slice(),
            wrap_width: self.wrap_width,
            max_lines: self.max_lines,
            force_width: self.force_width,
            font_generation: self.font_generation,
        }
    }
}

impl PartialEq for CacheKey {
    fn eq(&self, other: &Self) -> bool {
        self.as_cache_key_ref().eq(&other.as_cache_key_ref())
    }
}

impl Hash for CacheKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_cache_key_ref().hash(state);
    }
}

impl<'a> Borrow<dyn AsCacheKeyRef + 'a> for Arc<CacheKey> {
    fn borrow(&self) -> &(dyn AsCacheKeyRef + 'a) {
        self.as_ref() as &dyn AsCacheKeyRef
    }
}

impl AsCacheKeyRef for CacheKeyRef<'_> {
    fn as_cache_key_ref(&self) -> CacheKeyRef<'_> {
        *self
    }
}
