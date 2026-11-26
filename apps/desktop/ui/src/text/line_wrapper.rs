// Adapted from GPUI (Apache-2.0): crates/gpui/src/text_system/line_wrapper.rs
//! `LineWrapper`: GPUI's wrapping and truncation rules over per-character widths, which decide where
//! every line of the UI breaks. Copied rule for rule, quirks included.

use super::fx_hash::FxHashMap;
use crate::{FontId, FontRun, Pixels, PlatformTextSystem, SharedString, TextRun, px};
use std::borrow::Cow;
use std::iter;
use std::sync::Arc;
use unicode_segmentation::{Graphemes, UnicodeSegmentation};

/// Wraps and truncates lines of text in one font at one size, caching each character's width.
pub struct LineWrapper {
    platform_text_system: Arc<dyn PlatformTextSystem>,
    pub(crate) font_id: FontId,
    pub(crate) font_size: Pixels,
    cached_ascii_char_widths: [Option<Pixels>; 128],
    cached_other_char_widths: FxHashMap<char, Pixels>,
}

impl LineWrapper {
    /// The maximum indent that can be applied to a line.
    pub const MAX_INDENT: u32 = 256;

    pub(crate) fn new(font_id: FontId, font_size: Pixels, text_system: Arc<dyn PlatformTextSystem>) -> Self {
        Self {
            platform_text_system: text_system,
            font_id,
            font_size,
            cached_ascii_char_widths: [None; 128],
            cached_other_char_widths: FxHashMap::default(),
        }
    }

    /// The boundaries at which `fragments` wrap at `wrap_width`. A line breaks after an ASCII space
    /// before a word character, or before any other non-space, non-word grapheme cluster, never inside
    /// an extended grapheme cluster.
    pub fn wrap_line<'a>(
        &'a mut self,
        fragments: &'a [LineFragment],
        wrap_width: Pixels,
    ) -> impl Iterator<Item = Boundary> + 'a {
        let mut width = px(0.);
        let mut first_non_whitespace_ix = None;
        let mut indent = None;
        let mut last_candidate_ix = 0;
        let mut last_candidate_width = px(0.);
        let mut last_wrap_ix = 0;
        let mut prev_c = '\0';
        let mut index = 0;
        let mut candidates =
            fragments.iter().flat_map(move |fragment| fragment.wrap_boundary_candidates()).peekable();
        iter::from_fn(move || {
            for candidate in candidates.by_ref() {
                let ix = index;
                index += candidate.len_utf8();
                let mut new_prev_c = prev_c;
                let item_width = match candidate {
                    WrapBoundaryCandidate::Grapheme { grapheme } => {
                        if grapheme.ends_with('\n') {
                            continue;
                        }
                        let c = grapheme.chars().next().unwrap_or('\0');

                        if Self::is_word_char(c) {
                            if prev_c == ' ' && c != ' ' && first_non_whitespace_ix.is_some() {
                                last_candidate_ix = ix;
                                last_candidate_width = width;
                            }
                        } else {
                            // CJK may not be space separated, e.g.: `Hello world你好世界`
                            if c != ' ' && first_non_whitespace_ix.is_some() {
                                last_candidate_ix = ix;
                                last_candidate_width = width;
                            }
                        }

                        if c != ' ' && first_non_whitespace_ix.is_none() {
                            first_non_whitespace_ix = Some(ix);
                        }

                        new_prev_c = c;

                        self.width_for_grapheme(grapheme)
                    }
                    WrapBoundaryCandidate::Element { width: element_width, .. } => {
                        if prev_c == ' ' && first_non_whitespace_ix.is_some() {
                            last_candidate_ix = ix;
                            last_candidate_width = width;
                        }

                        if first_non_whitespace_ix.is_none() {
                            first_non_whitespace_ix = Some(ix);
                        }

                        element_width
                    }
                };

                width += item_width;
                if width > wrap_width && ix > last_wrap_ix {
                    if let (None, Some(first_non_whitespace_ix)) = (indent, first_non_whitespace_ix) {
                        indent = Some(Self::MAX_INDENT.min((first_non_whitespace_ix - last_wrap_ix) as u32));
                    }

                    if last_candidate_ix > 0 {
                        last_wrap_ix = last_candidate_ix;
                        width -= last_candidate_width;
                        last_candidate_ix = 0;
                    } else {
                        last_wrap_ix = ix;
                        width = item_width;
                    }

                    if let Some(indent) = indent {
                        width += self.width_for_char(' ') * indent as f32;
                    }

                    return Some(Boundary::new(last_wrap_ix, indent.unwrap_or(0)));
                }

                prev_c = new_prev_c;
            }

            None
        })
    }

    /// Truncates `line` to `truncate_width` with `truncation_suffix` (GPUI's rule: cut where the text
    /// plus the suffix still fits once the whole text would overflow `truncate_width`).
    ///
    /// Returns the text to lay out and runs that cover it exactly. The caller's `runs` are never
    /// changed: GPUI shortened them in place, so a later measure of the full text used lengths cut for
    /// the truncated one and sliced inside a multi-byte character.
    pub fn truncate_line(
        &mut self,
        line: SharedString,
        truncate_width: Pixels,
        truncation_suffix: &str,
        runs: &[TextRun],
    ) -> (SharedString, Vec<TextRun>) {
        let (result, truncated) = self.truncate_line_text(line, truncate_width, truncation_suffix);
        if truncated {
            let runs = runs_after_truncation(&result, truncation_suffix, runs);
            (result, runs)
        } else {
            (result, runs.to_vec())
        }
    }

    /// Truncates `line` when it overflows `truncate_width`, borrowing `runs` without allocating when
    /// the line already fits.
    pub(crate) fn truncate_line_cow<'a>(
        &mut self,
        line: SharedString,
        truncate_width: Pixels,
        truncation_suffix: &str,
        runs: &'a [TextRun],
    ) -> (SharedString, Cow<'a, [TextRun]>) {
        let (result, truncated) = self.truncate_line_text(line, truncate_width, truncation_suffix);
        if truncated {
            let runs = runs_after_truncation(&result, truncation_suffix, runs);
            (result, Cow::Owned(runs))
        } else {
            (result, Cow::Borrowed(runs))
        }
    }

    /// Truncates only the text string, returning `(text, was_truncated)` without touching runs.
    /// Cuts only on extended grapheme cluster boundaries so a combining mark is never separated from
    /// its base character.
    pub(crate) fn truncate_line_text(
        &mut self,
        line: SharedString,
        truncate_width: Pixels,
        truncation_suffix: &str,
    ) -> (SharedString, bool) {
        let mut width = px(0.);
        let suffix_width =
            truncation_suffix.graphemes(true).map(|g| self.width_for_grapheme(g)).fold(px(0.0), |a, x| a + x);
        let mut truncate_ix = 0;
        for (ix, grapheme) in line.grapheme_indices(true) {
            if width + suffix_width < truncate_width {
                truncate_ix = ix;
            }

            let grapheme_width = self.width_for_grapheme(grapheme);
            width += grapheme_width;

            if width.floor() > truncate_width {
                let mut result = String::with_capacity(truncate_ix + truncation_suffix.len());
                result.push_str(&line[..truncate_ix]);
                result.push_str(truncation_suffix);
                return (SharedString::from(result), true);
            }
        }

        (line, false)
    }

    /// Whether `c` is part of a word that should not be broken: ASCII alphanumerics, Latin-1
    /// Supplement, Latin Extended-A and -B, Cyrillic, `-_.'$%@#^~,=:` and `⋯`.
    pub fn is_word_char(c: char) -> bool {
        // ASCII alphanumeric characters, for English, numbers: `Hello123`, etc.
        c.is_ascii_alphanumeric() ||
        // Latin-1 Supplement
        matches!(c, '\u{00C0}'..='\u{00FF}') ||
        // Latin Extended-A
        matches!(c, '\u{0100}'..='\u{017F}') ||
        // Latin Extended-B
        matches!(c, '\u{0180}'..='\u{024F}') ||
        // Cyrillic for Russian, Ukrainian, etc.
        matches!(c, '\u{0400}'..='\u{04FF}') ||
        // `a-b`, `var_name`, `I'm`, `@mention`, `#hashtag`, `100%`, `3.1415`, `2^3`, `a~b`, `a=1`,
        // `Self::new`, etc.
        matches!(c, '-' | '_' | '.' | '\'' | '$' | '%' | '@' | '#' | '^' | '~' | ',' | '=' | ':') ||
        // `⋯` is kept at the end of the line.
        matches!(c, '⋯')
    }

    #[inline(always)]
    fn width_for_grapheme(&mut self, grapheme: &str) -> Pixels {
        if grapheme.len() == 1 {
            return self.width_for_char(grapheme.as_bytes()[0] as char);
        }
        let mut chars = grapheme.chars();
        if let Some(first) = chars.next()
            && chars.next().is_none()
        {
            return self.width_for_char(first);
        }
        self.platform_text_system
            .layout_line(grapheme, self.font_size, &[FontRun { len: grapheme.len(), font_id: self.font_id }])
            .width
    }

    #[inline(always)]
    fn width_for_char(&mut self, c: char) -> Pixels {
        if (c as u32) < 128 {
            if let Some(cached_width) = self.cached_ascii_char_widths[c as usize] {
                cached_width
            } else {
                let width = self.compute_width_for_char(c);
                self.cached_ascii_char_widths[c as usize] = Some(width);
                width
            }
        } else if let Some(cached_width) = self.cached_other_char_widths.get(&c) {
            *cached_width
        } else {
            let width = self.compute_width_for_char(c);
            self.cached_other_char_widths.insert(c, width);
            width
        }
    }

    fn compute_width_for_char(&self, c: char) -> Pixels {
        let mut buffer = [0; 4];
        let buffer = c.encode_utf8(&mut buffer);
        self.platform_text_system
            .layout_line(buffer, self.font_size, &[FontRun { len: buffer.len(), font_id: self.font_id }])
            .width
    }
}

/// Fresh runs for the truncated `result` (`runs` cut where the suffix starts, the run there extended by
/// the suffix and the rest dropped), GPUI's `update_runs_after_truncation` on a copy.
pub(crate) fn runs_after_truncation(result: &str, ellipsis: &str, runs: &[TextRun]) -> Vec<TextRun> {
    let mut runs = runs.to_vec();
    let mut truncate_at = result.len() - ellipsis.len();
    for (run_index, run) in runs.iter_mut().enumerate() {
        if run.len <= truncate_at {
            truncate_at -= run.len;
        } else {
            run.len = truncate_at + ellipsis.len();
            runs.truncate(run_index + 1);
            break;
        }
    }
    runs
}

/// A fragment of a line that can be wrapped.
pub enum LineFragment<'a> {
    /// A text fragment consisting of characters.
    Text {
        /// The text content of the fragment.
        text: &'a str,
    },
    /// A non-text element with a fixed width.
    Element {
        /// The width of the element in pixels.
        width: Pixels,
        /// The UTF-8 encoded length of the element.
        len_utf8: usize,
    },
}

impl<'a> LineFragment<'a> {
    /// A text fragment.
    pub fn text(text: &'a str) -> Self {
        LineFragment::Text { text }
    }

    /// A non-text element with the given width and UTF-8 encoded length.
    pub fn element(width: Pixels, len_utf8: usize) -> Self {
        LineFragment::Element { width, len_utf8 }
    }

    fn wrap_boundary_candidates(&self) -> WrapBoundaryCandidates<'a> {
        match self {
            LineFragment::Text { text } => WrapBoundaryCandidates::Text(text.graphemes(true)),
            LineFragment::Element { width, len_utf8 } => {
                WrapBoundaryCandidates::Element(Some((*width, *len_utf8)))
            }
        }
    }
}

enum WrapBoundaryCandidates<'a> {
    Text(Graphemes<'a>),
    Element(Option<(Pixels, usize)>),
}

impl<'a> Iterator for WrapBoundaryCandidates<'a> {
    type Item = WrapBoundaryCandidate<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            WrapBoundaryCandidates::Text(graphemes) => {
                graphemes.next().map(|grapheme| WrapBoundaryCandidate::Grapheme { grapheme })
            }
            WrapBoundaryCandidates::Element(item) => {
                item.take().map(|(width, len_utf8)| WrapBoundaryCandidate::Element { width, len_utf8 })
            }
        }
    }
}

enum WrapBoundaryCandidate<'a> {
    Grapheme { grapheme: &'a str },
    Element { width: Pixels, len_utf8: usize },
}

impl WrapBoundaryCandidate<'_> {
    fn len_utf8(&self) -> usize {
        match self {
            WrapBoundaryCandidate::Grapheme { grapheme } => grapheme.len(),
            WrapBoundaryCandidate::Element { len_utf8: len, .. } => *len,
        }
    }
}

/// A boundary between two lines of text.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Boundary {
    /// The index of the first character of the next line.
    pub ix: usize,
    /// The indent of the next line.
    pub next_indent: u32,
}

impl Boundary {
    /// A boundary at `ix` whose next line is indented by `next_indent`.
    pub fn new(ix: usize, next_indent: u32) -> Self {
        Self { ix, next_indent }
    }
}
