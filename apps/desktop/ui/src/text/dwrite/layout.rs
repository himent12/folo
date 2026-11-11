// Adapted from GPUI (Apache-2.0): crates/gpui/src/platform/windows/direct_write.rs (layout_line,
// TextRenderer, ClusterAnalyzer, StringIndexConverter, is_color_glyph)
//! Line layout through `IDWriteTextLayout` and a text renderer that collects the glyph runs: features
//! per run, fallback fonts given font ids on the fly, emoji detection, cluster-to-byte mapping.

use super::DirectWriteState;
use super::fonts::get_font_identifier_and_font_struct;
use crate::{FontRun, GlyphId, LineLayout, Pixels, ShapedGlyph, ShapedRun, point, px};
use anyhow::{Context as _, Result};
use std::ffi::c_void;
use windows::Win32::Foundation::E_NOTIMPL;
use windows::Win32::Graphics::DirectWrite::*;
use windows::core::{BOOL, HSTRING, IUnknown, Interface, Ref, implement};

/// Our `IDWriteTextRenderer` (GPUI's `TextRenderer`).
pub(super) fn text_renderer(locale: &str) -> IDWriteTextRenderer {
    TextRenderer { locale: locale.to_owned() }.into()
}

impl DirectWriteState {
    /// GPUI's `layout_line`: one text layout, the first run's font as its format, the other runs' fonts,
    /// sizes, styles, weights and features set on their UTF-16 ranges; ascent and descent from the
    /// first line's metrics (taken before the other runs are applied, as GPUI); glyphs from `Draw`.
    pub(super) fn layout_line(
        &mut self,
        text: &str,
        font_size: Pixels,
        font_runs: &[FontRun],
    ) -> Result<LineLayout> {
        let Some(first_run) = font_runs.first() else {
            return Ok(LineLayout { font_size, ..Default::default() });
        };
        let text_renderer = self.text_renderer.clone();
        let text_wide: Vec<u16> = text.encode_utf16().collect();

        let mut utf8_offset = 0usize;
        let mut utf16_offset = 0u32;
        let font_info = &self.fonts[first_run.font_id.0];
        let collection = self.collection_of(font_info).context("the font's collection is gone")?;
        // SAFETY: plain DirectWrite calls on live objects; ranges are within `text_wide`.
        let text_layout = unsafe {
            let format: IDWriteTextFormat1 = self
                .factory
                .CreateTextFormat(
                    &HSTRING::from(&font_info.font_family),
                    collection,
                    font_info.font_face.GetWeight(),
                    font_info.font_face.GetStyle(),
                    DWRITE_FONT_STRETCH_NORMAL,
                    font_size.0,
                    &HSTRING::from(&self.locale),
                )?
                .cast()?;
            if let Some(fallbacks) = &font_info.fallbacks {
                format.SetFontFallback(fallbacks)?;
            }
            let layout = self.factory.CreateTextLayout(&text_wide, &format, f32::INFINITY, f32::INFINITY)?;
            let current_text =
                text.get(utf8_offset..utf8_offset + first_run.len).context("run outside the text")?;
            utf8_offset += first_run.len;
            let current_text_utf16_length = current_text.encode_utf16().count() as u32;
            let text_range =
                DWRITE_TEXT_RANGE { startPosition: utf16_offset, length: current_text_utf16_length };
            layout.SetTypography(&font_info.features, text_range)?;
            utf16_offset += current_text_utf16_length;
            layout
        };

        let mut metrics = [DWRITE_LINE_METRICS::default(); 4];
        let mut line_count = 0u32;
        // SAFETY: `metrics` holds 4 lines; a single line of text has one.
        unsafe { text_layout.GetLineMetrics(Some(&mut metrics), &mut line_count)? };
        let ascent = px(metrics[0].baseline);
        let descent = px(metrics[0].height - metrics[0].baseline);

        for run in &font_runs[1..] {
            let font_info = &self.fonts[run.font_id.0];
            let current_text =
                text.get(utf8_offset..utf8_offset + run.len).context("run outside the text")?;
            utf8_offset += run.len;
            let current_text_utf16_length = current_text.encode_utf16().count() as u32;
            let collection = self.collection_of(font_info).context("the font's collection is gone")?;
            let text_range =
                DWRITE_TEXT_RANGE { startPosition: utf16_offset, length: current_text_utf16_length };
            utf16_offset += current_text_utf16_length;
            // SAFETY: as above.
            unsafe {
                text_layout.SetFontCollection(collection, text_range)?;
                text_layout.SetFontFamilyName(&HSTRING::from(&font_info.font_family), text_range)?;
                text_layout.SetFontSize(font_size.0, text_range)?;
                text_layout.SetFontStyle(font_info.font_face.GetStyle(), text_range)?;
                text_layout.SetFontWeight(font_info.font_face.GetWeight(), text_range)?;
                text_layout.SetTypography(&font_info.features, text_range)?;
            }
        }

        let mut runs = Vec::new();
        let mut context = RendererContext {
            state: self,
            index_converter: StringIndexConverter::new(text),
            runs: &mut runs,
            width: 0.0,
        };
        // SAFETY: `context` outlives the synchronous `Draw`, which hands it back to `DrawGlyphRun` only.
        unsafe {
            text_layout.Draw(
                Some(&mut context as *mut RendererContext as *const c_void),
                &text_renderer,
                0.0,
                0.0,
            )?
        };
        let width = px(context.width);

        Ok(LineLayout { font_size, width, ascent, descent, runs, len: text.len() })
    }
}

#[implement(IDWriteTextRenderer)]
struct TextRenderer {
    locale: String,
}

/// What `Draw` hands each `DrawGlyphRun`: the state (to map fallback faces to font ids), the
/// UTF-16 to UTF-8 index converter, the runs so far and the pen position.
struct RendererContext<'t, 'a, 'b> {
    state: &'t mut DirectWriteState,
    index_converter: StringIndexConverter<'a>,
    runs: &'b mut Vec<ShapedRun>,
    width: f32,
}

/// Splits a cluster map into (UTF-16 length, glyph count) per cluster.
#[derive(Debug)]
pub(super) struct ClusterAnalyzer<'t> {
    utf16_idx: usize,
    glyph_count: usize,
    cluster_map: &'t [u16],
}

impl<'t> ClusterAnalyzer<'t> {
    pub(super) fn new(cluster_map: &'t [u16], glyph_count: usize) -> Self {
        ClusterAnalyzer { utf16_idx: 0, glyph_count, cluster_map }
    }
}

impl Iterator for ClusterAnalyzer<'_> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<(usize, usize)> {
        if self.utf16_idx >= self.cluster_map.len() {
            return None;
        }
        let start_utf16_idx = self.utf16_idx;
        let current_glyph = self.cluster_map[start_utf16_idx] as usize;

        let mut end_utf16_idx = start_utf16_idx + 1;
        while end_utf16_idx < self.cluster_map.len()
            && self.cluster_map[end_utf16_idx] as usize == current_glyph
        {
            end_utf16_idx += 1;
        }
        let utf16_len = end_utf16_idx - start_utf16_idx;

        let next_glyph = if end_utf16_idx < self.cluster_map.len() {
            self.cluster_map[end_utf16_idx] as usize
        } else {
            self.glyph_count
        };
        let glyph_count = next_glyph.saturating_sub(current_glyph);

        self.utf16_idx = end_utf16_idx;
        Some((utf16_len, glyph_count))
    }
}

#[allow(non_snake_case)]
impl IDWritePixelSnapping_Impl for TextRenderer_Impl {
    fn IsPixelSnappingDisabled(&self, _clientdrawingcontext: *const c_void) -> windows::core::Result<BOOL> {
        Ok(BOOL(0))
    }

    fn GetCurrentTransform(
        &self,
        _clientdrawingcontext: *const c_void,
        transform: *mut DWRITE_MATRIX,
    ) -> windows::core::Result<()> {
        // SAFETY: DirectWrite passes a valid out pointer.
        unsafe { *transform = DWRITE_MATRIX { m11: 1.0, m12: 0.0, m21: 0.0, m22: 1.0, dx: 0.0, dy: 0.0 } };
        Ok(())
    }

    fn GetPixelsPerDip(&self, _clientdrawingcontext: *const c_void) -> windows::core::Result<f32> {
        Ok(1.0)
    }
}

#[allow(non_snake_case)]
impl IDWriteTextRenderer_Impl for TextRenderer_Impl {
    fn DrawGlyphRun(
        &self,
        clientdrawingcontext: *const c_void,
        _baselineoriginx: f32,
        _baselineoriginy: f32,
        _measuringmode: DWRITE_MEASURING_MODE,
        glyphrun: *const DWRITE_GLYPH_RUN,
        glyphrundescription: *const DWRITE_GLYPH_RUN_DESCRIPTION,
        _clientdrawingeffect: Ref<IUnknown>,
    ) -> windows::core::Result<()> {
        if glyphrun.is_null() || glyphrundescription.is_null() || clientdrawingcontext.is_null() {
            return Ok(());
        }
        // SAFETY: DirectWrite passes a valid glyph run for the duration of the call (not null: above).
        let glyphrun = unsafe { &*glyphrun };
        let glyph_count = glyphrun.glyphCount as usize;
        let Some(font_face) = glyphrun.fontFace.as_ref() else {
            return Ok(());
        };
        if glyph_count == 0 {
            return Ok(());
        }
        // SAFETY: as above, for the description.
        let desc = unsafe { &*glyphrundescription };
        // The glyphs, their advances and the cluster map are needed; offsets may be absent (all zero).
        if glyphrun.glyphIndices.is_null() || glyphrun.glyphAdvances.is_null() || desc.clusterMap.is_null() {
            return Ok(());
        }
        // SAFETY: `clientdrawingcontext` is the `RendererContext` `layout_line` passed to `Draw`, alive and
        // not otherwise borrowed while `Draw` runs.
        let context = unsafe { &mut *(clientdrawingcontext as *mut RendererContext) };
        // `IDWriteFontFace3` exists on Windows 10+.
        let font_face: IDWriteFontFace3 = font_face.cast()?;
        let Some((font_identifier, font_struct, color_font)) =
            get_font_identifier_and_font_struct(&font_face, &self.locale)
        else {
            return Ok(());
        };

        let font_id = match context.state.font_id_by_identifier.get(&font_identifier) {
            Some(id) => *id,
            None => match context.state.select_font(&font_struct) {
                Ok(id) => id,
                Err(_) => return Ok(()),
            },
        };

        // SAFETY: DirectWrite's arrays hold `glyph_count` glyphs and `stringLength` cluster entries, and
        // the pointers are not null (checked above; absent offsets give an empty slice).
        let (glyph_ids, glyph_advances, glyph_offsets, cluster_map) = unsafe {
            (
                std::slice::from_raw_parts(glyphrun.glyphIndices, glyph_count),
                std::slice::from_raw_parts(glyphrun.glyphAdvances, glyph_count),
                if glyphrun.glyphOffsets.is_null() {
                    &[]
                } else {
                    std::slice::from_raw_parts(glyphrun.glyphOffsets, glyph_count)
                },
                std::slice::from_raw_parts(desc.clusterMap, desc.stringLength as usize),
            )
        };

        let advance_offset = |i: usize| glyph_offsets.get(i).map_or(0.0, |o| o.advanceOffset);
        let cluster_analyzer = ClusterAnalyzer::new(cluster_map, glyph_count);
        let mut utf16_idx = desc.textPosition as usize;
        let mut glyph_idx = 0;
        let mut glyphs = Vec::with_capacity(glyph_count);
        for (cluster_utf16_len, cluster_glyph_count) in cluster_analyzer {
            context.index_converter.advance_to_utf16_ix(utf16_idx);
            utf16_idx += cluster_utf16_len;
            let end = (glyph_idx + cluster_glyph_count).min(glyph_count);
            for (cluster_glyph_idx, glyph_id) in glyph_ids[glyph_idx..end].iter().enumerate() {
                let id = GlyphId(*glyph_id as u32);
                let is_emoji = color_font && is_color_glyph(&font_face, id, &context.state.factory);
                let this_glyph_idx = glyph_idx + cluster_glyph_idx;
                glyphs.push(ShapedGlyph {
                    id,
                    position: point(px(context.width + advance_offset(this_glyph_idx)), px(0.0)),
                    index: context.index_converter.utf8_ix,
                    is_emoji,
                });
                context.width += glyph_advances[this_glyph_idx];
            }
            glyph_idx = end;
        }
        context.runs.push(ShapedRun { font_id, glyphs });
        Ok(())
    }

    fn DrawUnderline(
        &self,
        _clientdrawingcontext: *const c_void,
        _baselineoriginx: f32,
        _baselineoriginy: f32,
        _underline: *const DWRITE_UNDERLINE,
        _clientdrawingeffect: Ref<IUnknown>,
    ) -> windows::core::Result<()> {
        Err(windows::core::Error::new(E_NOTIMPL, "DrawUnderline unimplemented"))
    }

    fn DrawStrikethrough(
        &self,
        _clientdrawingcontext: *const c_void,
        _baselineoriginx: f32,
        _baselineoriginy: f32,
        _strikethrough: *const DWRITE_STRIKETHROUGH,
        _clientdrawingeffect: Ref<IUnknown>,
    ) -> windows::core::Result<()> {
        Err(windows::core::Error::new(E_NOTIMPL, "DrawStrikethrough unimplemented"))
    }

    fn DrawInlineObject(
        &self,
        _clientdrawingcontext: *const c_void,
        _originx: f32,
        _originy: f32,
        _inlineobject: Ref<IDWriteInlineObject>,
        _issideways: BOOL,
        _isrighttoleft: BOOL,
        _clientdrawingeffect: Ref<IUnknown>,
    ) -> windows::core::Result<()> {
        Err(windows::core::Error::new(E_NOTIMPL, "DrawInlineObject unimplemented"))
    }
}

/// Walks a UTF-8 string by UTF-16 offsets.
pub(super) struct StringIndexConverter<'a> {
    text: &'a str,
    pub(super) utf8_ix: usize,
    utf16_ix: usize,
}

impl<'a> StringIndexConverter<'a> {
    pub(super) fn new(text: &'a str) -> Self {
        Self { text, utf8_ix: 0, utf16_ix: 0 }
    }

    /// Moves to the first character at or after the UTF-16 offset `utf16_target`.
    pub(super) fn advance_to_utf16_ix(&mut self, utf16_target: usize) {
        for (ix, c) in self.text[self.utf8_ix..].char_indices() {
            if self.utf16_ix >= utf16_target {
                self.utf8_ix += ix;
                return;
            }
            self.utf16_ix += c.len_utf16();
        }
        self.utf8_ix = self.text.len();
    }
}

/// Whether the glyph has a colour image in any format (GPUI's check: `IDWriteFontFace4`'s
/// `GetGlyphImageFormats` misses some glyphs, say ❤).
fn is_color_glyph(font_face: &IDWriteFontFace3, glyph_id: GlyphId, factory: &IDWriteFactory5) -> bool {
    let index = glyph_id.0 as u16;
    let advance = 0.0f32;
    let offset = DWRITE_GLYPH_OFFSET { advanceOffset: 0.0, ascenderOffset: 0.0 };
    let glyph_run = DWRITE_GLYPH_RUN {
        // SAFETY: a borrowed copy of the face pointer, never released (`ManuallyDrop`).
        fontFace: unsafe { std::mem::transmute_copy(font_face) },
        fontEmSize: 14.0,
        glyphCount: 1,
        glyphIndices: &index,
        glyphAdvances: &advance,
        glyphOffsets: &offset,
        isSideways: BOOL(0),
        bidiLevel: 0,
    };
    // SAFETY: the run's arrays live on this stack frame for the call.
    unsafe {
        factory.TranslateColorGlyphRun(
            super::raster::vector2(0.0, 0.0),
            &glyph_run,
            None,
            DWRITE_GLYPH_IMAGE_FORMATS_COLR
                | DWRITE_GLYPH_IMAGE_FORMATS_SVG
                | DWRITE_GLYPH_IMAGE_FORMATS_PNG
                | DWRITE_GLYPH_IMAGE_FORMATS_JPEG
                | DWRITE_GLYPH_IMAGE_FORMATS_PREMULTIPLIED_B8G8R8A8,
            DWRITE_MEASURING_MODE_NATURAL,
            None,
            0,
        )
    }
    .is_ok()
}

#[cfg(test)]
mod tests {
    use super::{ClusterAnalyzer, StringIndexConverter};

    /// GPUI's `test_cluster_map`.
    #[test]
    fn cluster_map() {
        let analyze = |map: &[u16], glyphs| ClusterAnalyzer::new(map, glyphs).collect::<Vec<_>>();
        assert_eq!(analyze(&[0], 1), [(1, 1)]);
        assert_eq!(analyze(&[0, 1, 2], 3), [(1, 1), (1, 1), (1, 1)]);
        // 👨‍👩‍👧‍👦👩‍💻
        assert_eq!(analyze(&[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 4], 5), [(11, 4), (5, 1)]);
        // 👩‍💻
        assert_eq!(analyze(&[0, 0, 0, 0, 0], 1), [(5, 1)]);
    }

    #[test]
    fn utf16_offsets_map_to_utf8_boundaries() {
        let text = "a·é😀b";
        let mut converter = StringIndexConverter::new(text);
        let mut seen = Vec::new();
        // UTF-16 offsets of each character: a=0, ·=1, é=2, 😀=3 (two units), b=5.
        for target in [0, 1, 2, 3, 5] {
            converter.advance_to_utf16_ix(target);
            seen.push(converter.utf8_ix);
        }
        assert_eq!(seen, [0, 1, 3, 5, 9]);
        converter.advance_to_utf16_ix(6);
        assert_eq!(converter.utf8_ix, text.len());
    }
}
