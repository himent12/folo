// Adapted from GPUI (Apache-2.0): crates/gpui/src/text_system/line_wrapper.rs
//! `LineWrapper`: GPUI's wrapping and truncation rules over per-character widths, which decide where
//! every line of the UI breaks. Copied rule for rule, quirks included.

use crate::{FontId, FontRun, Pixels, PlatformTextSystem, SharedString, TextRun, px};
use std::collections::HashMap;
use std::iter;
use std::sync::Arc;

/// Wraps and truncates lines of text in one font at one size, caching each character's width.
pub struct LineWrapper {
    platform_text_system: Arc<dyn PlatformTextSystem>,
    pub(crate) font_id: FontId,
    pub(crate) font_size: Pixels,
    cached_ascii_char_widths: [Option<Pixels>; 128],
    cached_other_char_widths: HashMap<char, Pixels>,
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
            cached_other_char_widths: HashMap::default(),
        }
    }

    /// The boundaries at which `fragments` wrap at `wrap_width`. A line breaks after an ASCII space
    /// before a word character, or before any other non-space, non-word character.
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
                    WrapBoundaryCandidate::Char { character: c } => {
                        if c == '\n' {
                            continue;
                        }

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

                        self.width_for_char(c)
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
        let mut width = px(0.);
        let suffix_width =
            truncation_suffix.chars().map(|c| self.width_for_char(c)).fold(px(0.0), |a, x| a + x);
        let mut truncate_ix = 0;
        for (ix, c) in line.char_indices() {
            if width + suffix_width < truncate_width {
                truncate_ix = ix;
            }

            let char_width = self.width_for_char(c);
            width += char_width;

            if width.floor() > truncate_width {
                let result = SharedString::from(format!("{}{}", &line[..truncate_ix], truncation_suffix));
                let runs = runs_after_truncation(&result, truncation_suffix, runs);
                return (result, runs);
            }
        }

        (line, runs.to_vec())
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

    fn wrap_boundary_candidates(&self) -> impl Iterator<Item = WrapBoundaryCandidate> {
        let text = match self {
            LineFragment::Text { text } => text,
            LineFragment::Element { .. } => "\0",
        };
        text.chars().map(move |character| {
            if let LineFragment::Element { width, len_utf8 } = self {
                WrapBoundaryCandidate::Element { width: *width, len_utf8: *len_utf8 }
            } else {
                WrapBoundaryCandidate::Char { character }
            }
        })
    }
}

enum WrapBoundaryCandidate {
    Char { character: char },
    Element { width: Pixels, len_utf8: usize },
}

impl WrapBoundaryCandidate {
    fn len_utf8(&self) -> usize {
        match self {
            WrapBoundaryCandidate::Char { character } => character.len_utf8(),
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
