// Adapted from GPUI (Apache-2.0): crates/gpui/src/text_system.rs (WindowTextSystem)
//! `WindowTextSystem`: a window's view of the text system, with the line layout cache that keeps what
//! the previous frame used.

use super::layout_cache::{LineLayoutCache, LineLayoutIndex};
use crate::{
    DecorationRun, FontId, FontRun, LineLayout, Pixels, ShapedLine, SharedString, TextRun, TextSystem,
    WrappedLine,
};
use anyhow::Result;
use smallvec::SmallVec;
use std::cmp;
use std::ops::{Deref, Range};
use std::sync::Arc;

/// A window's text system: shapes lines through the per-window layout cache (derefs to `TextSystem`).
pub struct WindowTextSystem {
    line_layout_cache: LineLayoutCache,
    text_system: Arc<TextSystem>,
}

impl Deref for WindowTextSystem {
    type Target = Arc<TextSystem>;

    fn deref(&self) -> &Self::Target {
        &self.text_system
    }
}

impl WindowTextSystem {
    /// A window text system over the app's text system, with an empty layout cache.
    pub fn new(text_system: Arc<TextSystem>) -> Self {
        Self {
            line_layout_cache: LineLayoutCache::new(text_system.platform_text_system.clone()),
            text_system,
        }
    }

    /// The position in the current frame's used layouts (the window marks a cached element's range).
    pub fn layout_index(&self) -> LineLayoutIndex {
        self.line_layout_cache.layout_index()
    }

    /// Carries the previous frame's layouts in `index` over to this frame (a reused element).
    pub fn reuse_layouts(&self, index: Range<LineLayoutIndex>) {
        self.line_layout_cache.reuse_layouts(index)
    }

    /// Forgets the layouts this frame used after `index`.
    pub fn truncate_layouts(&self, index: LineLayoutIndex) {
        self.line_layout_cache.truncate_layouts(index)
    }

    /// How many lines this window has asked the platform to shape (cache misses).
    pub fn shaped_line_count(&self) -> usize {
        self.line_layout_cache.shaped_lines()
    }

    /// Shapes one line (no newlines) at `font_size`, styled by `runs`, ready to paint.
    pub fn shape_line(
        &self,
        text: SharedString,
        font_size: Pixels,
        runs: &[TextRun],
        force_width: Option<Pixels>,
    ) -> ShapedLine {
        debug_assert!(text.find('\n').is_none(), "text argument should not contain newlines");

        let mut decoration_runs = SmallVec::<[DecorationRun; 2]>::new();
        for run in runs {
            if let Some(last_run) = decoration_runs.last_mut()
                && last_run.color == run.color
                && last_run.underline == run.underline
                && last_run.strikethrough == run.strikethrough
                && last_run.background_color == run.background_color
            {
                last_run.len += run.len as u32;
                continue;
            }
            decoration_runs.push(DecorationRun {
                len: run.len as u32,
                color: run.color,
                background_color: run.background_color,
                underline: run.underline,
                strikethrough: run.strikethrough,
            });
        }

        let layout = self.layout_line_with(&text, || text.clone(), font_size, runs, force_width);

        ShapedLine { layout, text, decoration_runs }
    }

    /// Shapes text with newlines into one `WrappedLine` per line, wrapped at `wrap_width` and limited
    /// to `line_clamp` visual lines.
    pub fn shape_text(
        &self,
        text: SharedString,
        font_size: Pixels,
        runs: &[TextRun],
        wrap_width: Option<Pixels>,
        line_clamp: Option<usize>,
    ) -> Result<SmallVec<[WrappedLine; 1]>> {
        struct RunCursor<'a> {
            runs: &'a [TextRun],
            index: usize,
            remaining: usize,
        }
        impl<'a> RunCursor<'a> {
            fn new(runs: &'a [TextRun]) -> Self {
                let mut cursor = Self { runs, index: 0, remaining: 0 };
                cursor.advance_to_nonempty();
                cursor
            }
            fn advance_to_nonempty(&mut self) {
                while self.index < self.runs.len() {
                    let len = self.runs[self.index].len;
                    if len > 0 {
                        self.remaining = len;
                        return;
                    }
                    self.index += 1;
                }
                self.remaining = 0;
            }
            fn current(&self) -> Option<(usize, &'a TextRun, usize)> {
                (self.remaining > 0).then(|| (self.index, &self.runs[self.index], self.remaining))
            }
            fn consume(&mut self, amount: usize) {
                debug_assert!(amount <= self.remaining);
                self.remaining -= amount;
                if self.remaining == 0 {
                    self.index += 1;
                    self.advance_to_nonempty();
                }
            }
        }

        let mut cursor = RunCursor::new(runs);
        let mut last_resolved: Option<(usize, FontId)> = None;
        let mut font_runs = SmallVec::<[FontRun; 4]>::new();
        let font_generation = self.text_system.font_generation();

        let mut lines = SmallVec::new();
        let mut line_start = 0;
        let mut wrapped_lines = 0;

        let mut process_line = |line_text: SharedString| {
            if line_clamp.is_some_and(|max_lines| wrapped_lines >= max_lines) {
                return;
            }
            font_runs.clear();
            let line_end = line_start + line_text.len();

            let mut decoration_runs = SmallVec::<[DecorationRun; 2]>::new();
            let mut run_start = line_start;
            while run_start < line_end {
                let Some((run_ix, run, remaining)) = cursor.current() else {
                    break;
                };

                let run_len_within_line = cmp::min(line_end - run_start, remaining);

                let decoration_changed = if let Some(last_run) = decoration_runs.last_mut()
                    && last_run.color == run.color
                    && last_run.underline == run.underline
                    && last_run.strikethrough == run.strikethrough
                    && last_run.background_color == run.background_color
                {
                    last_run.len += run_len_within_line as u32;
                    false
                } else {
                    decoration_runs.push(DecorationRun {
                        len: run_len_within_line as u32,
                        color: run.color,
                        background_color: run.background_color,
                        underline: run.underline,
                        strikethrough: run.strikethrough,
                    });
                    true
                };

                let font_id = match last_resolved {
                    Some((idx, id)) if idx == run_ix => id,
                    _ => {
                        let id = self.resolve_font(&run.font);
                        last_resolved = Some((run_ix, id));
                        id
                    }
                };
                if let Some(font_run) = font_runs.last_mut()
                    && font_id == font_run.font_id
                    && !decoration_changed
                {
                    font_run.len += run_len_within_line;
                } else {
                    font_runs.push(FontRun { len: run_len_within_line, font_id });
                }

                cursor.consume(run_len_within_line);
                run_start += run_len_within_line;
            }

            let layout = self.line_layout_cache.layout_wrapped_line(
                &line_text,
                || line_text.clone(),
                font_size,
                &font_runs,
                wrap_width,
                // The lines this one may still use (GPUI passes `usize::MAX` less the lines so far
                // when there is no clamp, which limits nothing).
                line_clamp.map(|max_lines| max_lines.saturating_sub(wrapped_lines)),
                font_generation,
            );
            wrapped_lines += layout.wrap_boundaries.len() + 1;

            lines.push(WrappedLine { layout, decoration_runs, text: line_text });

            // Skip the `\n`.
            line_start = line_end + 1;
            if cursor.current().is_some() {
                cursor.consume(1);
            }
        };

        let mut split_lines = text.split('\n');
        let mut processed = false;

        if let Some(first_line) = split_lines.next()
            && let Some(second_line) = split_lines.next()
        {
            processed = true;
            process_line(first_line.to_string().into());
            process_line(second_line.to_string().into());
            for line_text in split_lines {
                process_line(line_text.to_string().into());
            }
        }

        if !processed {
            process_line(text);
        }

        Ok(lines)
    }

    /// Single-style fast path for plain text elements: avoids allocating a `Vec<TextRun>` or cloning
    /// `Font` while producing the exact same `WrappedLine`s as `shape_text` with `[style.to_run(text.len())]`.
    pub(crate) fn shape_text_with_style(
        &self,
        text: SharedString,
        font_size: Pixels,
        style: &crate::TextStyle,
        wrap_width: Option<Pixels>,
        line_clamp: Option<usize>,
    ) -> Result<SmallVec<[WrappedLine; 1]>> {
        let font_generation = self.text_system.font_generation();
        let mut font_id: Option<FontId> = None;

        let mut lines = SmallVec::new();
        let mut wrapped_lines = 0;

        let mut process_line = |line_text: SharedString| {
            if line_clamp.is_some_and(|max_lines| wrapped_lines >= max_lines) {
                return;
            }
            let mut decoration_runs = SmallVec::<[DecorationRun; 2]>::new();
            let mut font_runs = SmallVec::<[FontRun; 1]>::new();
            if !line_text.is_empty() {
                let font_id = *font_id.get_or_insert_with(|| self.resolve_font_key(style.font_key_ref()));
                decoration_runs.push(DecorationRun {
                    len: line_text.len() as u32,
                    color: style.color,
                    background_color: style.background_color,
                    underline: style.underline,
                    strikethrough: style.strikethrough,
                });
                font_runs.push(FontRun { len: line_text.len(), font_id });
            }

            let layout = self.line_layout_cache.layout_wrapped_line(
                &line_text,
                || line_text.clone(),
                font_size,
                &font_runs,
                wrap_width,
                line_clamp.map(|max_lines| max_lines.saturating_sub(wrapped_lines)),
                font_generation,
            );
            wrapped_lines += layout.wrap_boundaries.len() + 1;

            lines.push(WrappedLine { layout, decoration_runs, text: line_text });
        };

        let mut split_lines = text.split('\n');
        let mut processed = false;

        if let Some(first_line) = split_lines.next()
            && let Some(second_line) = split_lines.next()
        {
            processed = true;
            process_line(first_line.to_string().into());
            process_line(second_line.to_string().into());
            for line_text in split_lines {
                process_line(line_text.to_string().into());
            }
        }

        if !processed {
            process_line(text);
        }

        Ok(lines)
    }

    /// Ends the frame: layouts used this frame become the "previous frame" the next one reuses.
    pub fn finish_frame(&self) {
        self.line_layout_cache.finish_frame()
    }

    /// Lays out one line at `font_size`, styled by `runs` (prefer `shape_line`, which can be painted).
    pub fn layout_line(
        &self,
        text: &str,
        font_size: Pixels,
        runs: &[TextRun],
        force_width: Option<Pixels>,
    ) -> Arc<LineLayout> {
        self.layout_line_with(text, || SharedString::new(text), font_size, runs, force_width)
    }

    fn layout_line_with(
        &self,
        text: &str,
        into_shared: impl FnOnce() -> SharedString,
        font_size: Pixels,
        runs: &[TextRun],
        force_width: Option<Pixels>,
    ) -> Arc<LineLayout> {
        let mut last_run = None::<&TextRun>;
        let mut last_font: Option<FontId> = None;
        let mut font_runs = SmallVec::<[FontRun; 4]>::new();
        let font_generation = self.text_system.font_generation();

        for run in runs.iter() {
            let decoration_changed = if let Some(last_run) = last_run
                && last_run.color == run.color
                && last_run.underline == run.underline
                && last_run.strikethrough == run.strikethrough
            // GPUI ignores the background colour here: it does not affect glyphs.
            {
                false
            } else {
                last_run = Some(run);
                true
            };

            if let Some(font_run) = font_runs.last_mut()
                && Some(font_run.font_id) == last_font
                && !decoration_changed
            {
                font_run.len += run.len;
            } else {
                let font_id = self.resolve_font(&run.font);
                last_font = Some(font_id);
                font_runs.push(FontRun { len: run.len, font_id });
            }
        }

        self.line_layout_cache.layout_line(
            text,
            into_shared,
            font_size,
            &font_runs,
            force_width,
            font_generation,
        )
    }
}
