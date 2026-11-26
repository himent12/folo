// Adapted from GPUI (Apache-2.0): crates/gpui/src/platform/linux/text_system.rs
//! `layout_line`: one line shaped by cosmic-text (GPUI's Linux backend, plus the lazy system-font load
//! when a glyph is missing from the fonts the database holds).

use super::fonts::State;
use crate::{FontId, FontRun, GlyphId, LineLayout, Pixels, ShapedGlyph, ShapedRun, point};
use cosmic_text::{Attrs, AttrsList, Family, LayoutLine, ShapeLine, Shaping, Wrap};

impl State {
    pub fn layout_line(&mut self, text: &str, font_size: Pixels, font_runs: &[FontRun]) -> LineLayout {
        let mut line = self.shape(text, font_size, font_runs);
        if !self.system_fonts_loaded() && has_missing_glyphs(text, line.as_ref()) {
            self.load_system_fonts();
            line = self.shape(text, font_size, font_runs);
        }
        let Some(line) = line else {
            return LineLayout { font_size, len: text.len(), ..Default::default() };
        };

        let mut runs: Vec<ShapedRun> = Vec::new();
        for glyph in &line.glyphs {
            let mut font_id = FontId(glyph.metadata);
            if self.loaded_font(font_id).font.id() != glyph.font_id {
                font_id = self.font_id_for_cosmic_id(glyph.font_id);
            }
            let is_emoji = self.loaded_font(font_id).is_known_emoji_font;

            // GPUI's guard: variation selectors shape to glyph 3 in emoji fonts; drop them.
            if glyph.glyph_id == 3 && is_emoji {
                continue;
            }

            let shaped = ShapedGlyph {
                id: GlyphId(glyph.glyph_id as u32),
                position: point(glyph.x.into(), glyph.y.into()),
                index: glyph.start,
                is_emoji,
            };
            match runs.last_mut().filter(|run| run.font_id == font_id) {
                Some(run) => run.glyphs.push(shaped),
                None => runs.push(ShapedRun { font_id, glyphs: vec![shaped] }),
            }
        }

        LineLayout {
            font_size,
            width: line.w.into(),
            ascent: line.max_ascent.into(),
            descent: line.max_descent.into(),
            runs,
            len: text.len(),
        }
    }

    /// Shapes and lays out `text` as one unwrapped line in LTR paragraph base direction (matching
    /// DirectWrite's `DWRITE_READING_DIRECTION_LEFT_TO_RIGHT`). `None` only if cosmic-text returns no
    /// line.
    fn shape(&mut self, text: &str, font_size: Pixels, font_runs: &[FontRun]) -> Option<LayoutLine> {
        let attrs_list = self.attrs_list(font_runs, 0);
        let shaped = ShapeLine::new(&mut self.font_system, text, &attrs_list, Shaping::Advanced, 4);
        if !shaped.rtl {
            let mut lines = Vec::with_capacity(1);
            shaped.layout_to_buffer(&mut self.scratch, font_size.0, None, Wrap::None, None, &mut lines, None);
            return lines.into_iter().next();
        }

        // cosmic-text auto-detects RTL paragraph direction from the first strong character, which
        // lays out glyphs right-to-left (decreasing x, reversed logical wrapping order, and -epsilon
        // rounding at the left edge). Re-shape with a temporary leading LRM so the paragraph base
        // direction is LTR (while RTL spans inside it still shape right-to-left), then strip the LRM.
        let lrm_len = '\u{200E}'.len_utf8();
        let mut prefixed = String::with_capacity(lrm_len + text.len());
        prefixed.push('\u{200E}');
        prefixed.push_str(text);
        let ltr_attrs = self.attrs_list(font_runs, lrm_len);
        let shaped = ShapeLine::new(&mut self.font_system, &prefixed, &ltr_attrs, Shaping::Advanced, 4);
        let mut lines = Vec::with_capacity(1);
        shaped.layout_to_buffer(&mut self.scratch, font_size.0, None, Wrap::None, None, &mut lines, None);
        let mut line = lines.into_iter().next()?;
        let mut dropped_w = 0.0f32;
        line.glyphs.retain_mut(|g| {
            if g.end <= lrm_len {
                dropped_w += g.w;
                false
            } else {
                g.start = g.start.saturating_sub(lrm_len);
                g.end -= lrm_len;
                g.x = (g.x - dropped_w).max(0.0);
                true
            }
        });
        line.w = (line.w - dropped_w).max(0.0);
        Some(line)
    }

    fn attrs_list(&self, font_runs: &[FontRun], leading_prefix_len: usize) -> AttrsList {
        let mut attrs_list = AttrsList::new(&Attrs::new());
        let mut offset = 0;
        for (i, run) in font_runs.iter().enumerate() {
            let loaded = self.loaded_font(run.font_id);
            let face =
                self.font_system.db().face(loaded.font.id()).expect("a loaded face is in the database");
            let len = if i == 0 { run.len + leading_prefix_len } else { run.len };
            attrs_list.add_span(
                offset..(offset + len),
                &Attrs::new()
                    .metadata(run.font_id.0)
                    .family(Family::Name(&face.families.first().expect("a face has a family").0))
                    .stretch(face.stretch)
                    .style(face.style)
                    .weight(face.weight)
                    .font_features(loaded.features.clone()),
            );
            offset += len;
        }
        attrs_list
    }
}

/// Whether a visible character of `text` was shaped to the missing glyph (id 0) in every font tried.
fn has_missing_glyphs(text: &str, line: Option<&LayoutLine>) -> bool {
    let Some(line) = line else { return false };
    line.glyphs.iter().any(|glyph| {
        glyph.glyph_id == 0
            && text
                .get(glyph.start..glyph.end)
                .is_some_and(|s| s.chars().any(|c| !c.is_whitespace() && !c.is_control()))
    })
}
