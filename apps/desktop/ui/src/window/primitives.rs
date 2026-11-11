//! The element-author paint API on `Window` (GPUI's signatures): quads, paths, images, SVG icons, glyphs,
//! emoji, shadows, underlines. Owner tag: `"window"`.
//!
//! Every paint call scales to device pixels where GPUI scales, rounds where GPUI rounds, takes the current
//! content mask (scaled) as the primitive's clip, fades colours by the element opacity, skips a primitive
//! whose bounds miss the clip (GPUI's `Scene::insert_primitive`) and pushes it onto `next_frame.scene`
//! (painter's order). Quads, shadows and paths go through the style package's pure functions
//! (`style/paint.rs`, `path_builder.rs`), which do exactly that. Glyph and icon masks come from the window's
//! `MaskCache`s (keyed by `RenderGlyphParams` / `RenderSvgParams`), so steady frames rasterize nothing;
//! images are shared as `Arc<RenderImage>`.

use std::sync::Arc;

use crate::{
    AlphaMask, App, Background, Bounds, BoxShadow, BuiltPath, ColorMask, ColorSprite, Corners, FontId,
    GlyphId, Hsla, Image, MonoSprite, PaintQuad, Pixels, Point, RenderGlyphParams, RenderImage,
    RenderSvgParams, SUBPIXEL_VARIANTS_X, SUBPIXEL_VARIANTS_Y, ScaledPixels, SharedString, Size, SpriteKind,
    StrikethroughStyle, StylePaint, Underline, UnderlineStyle, Window, raster::Mask, size, svg_sprite_bounds,
};

impl Window {
    /// The window state the style package's paint functions read (rem size, scale, element opacity, mask).
    pub(crate) fn style_paint(&self) -> StylePaint {
        StylePaint {
            rem_size: self.rem_size(),
            scale_factor: self.scale_factor,
            opacity: self.element_opacity,
            content_mask: self.content_mask(),
        }
    }

    /// The current content mask in device pixels: every primitive's clip.
    fn scaled_clip(&self) -> Bounds<ScaledPixels> {
        self.content_mask().bounds.scale(self.scale_factor)
    }

    /// Paints a quad (GPUI's `paint_quad`: scaled, not rounded; radii may exceed the bounds).
    pub fn paint_quad(&mut self, quad: PaintQuad) {
        let cx = self.style_paint();
        crate::paint_quad(
            &mut self.next_frame.scene,
            &cx,
            quad.bounds,
            quad.corner_radii,
            quad.background,
            quad.border_widths,
            quad.border_color,
            quad.border_style,
        );
    }

    /// Paints a path built by `PathBuilder` (fill or stroke) with a colour or gradient.
    pub fn paint_path(&mut self, path: BuiltPath, color: impl Into<Background>) {
        let cx = self.style_paint();
        crate::paint_path(&mut self.next_frame.scene, &cx, &path, color);
    }

    /// Paints frame `frame_index` of an image (origin floored, size ceiled in device pixels, as GPUI).
    pub fn paint_image(
        &mut self,
        bounds: Bounds<Pixels>,
        corner_radii: Corners<Pixels>,
        data: Arc<RenderImage>,
        frame_index: usize,
        grayscale: bool,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(frame_index < data.frame_count(), "image frame {frame_index} out of range");
        let scale_factor = self.scale_factor;
        let bounds =
            bounds.scale(scale_factor).map_origin(|origin| origin.floor()).map_size(|size| size.ceil());
        let clip = self.scaled_clip();
        if bounds.intersect(&clip).is_empty() {
            return Ok(());
        }
        self.next_frame.scene.push_image(Image {
            bounds,
            clip,
            corner_radii: corner_radii.scale(scale_factor),
            image: data,
            frame: frame_index,
            grayscale,
            opacity: self.element_opacity,
        });
        Ok(())
    }

    /// Paints an SVG asset as a single-colour icon (mask from the icon cache, rendered at GPUI's
    /// `SMOOTH_SVG_SCALE_FACTOR` and placed by `svg_sprite_bounds`).
    pub fn paint_svg(
        &mut self,
        bounds: Bounds<Pixels>,
        path: SharedString,
        color: Hsla,
        _cx: &App,
    ) -> anyhow::Result<()> {
        let bounds = bounds.scale(self.scale_factor);
        let params = RenderSvgParams::for_bounds(path, &bounds);
        let renderer = &self.svg_renderer;
        let mask = self.icon_masks.get_or_insert_with(&params, || {
            Ok(renderer.render(&params)?.map(|mask| Mask::from(Arc::new(mask))))
        })?;
        let Some(mask) = mask.and_then(|mask| mask.alpha().cloned()) else { return Ok(()) };
        let bounds = svg_sprite_bounds(bounds, mask.size);
        let clip = self.scaled_clip();
        if bounds.intersect(&clip).is_empty() {
            return Ok(());
        }
        let color = color.opacity(self.element_opacity);
        self.next_frame.scene.push_mono_sprite(MonoSprite {
            bounds,
            clip,
            color,
            mask,
            kind: SpriteKind::Svg,
        });
        Ok(())
    }

    /// Paints one glyph whose baseline origin is `origin` (GPUI's `paint_glyph`, as the text package
    /// specifies: subpixel variant from the scaled origin's fraction, raster bounds from the text system,
    /// the mask cached by its `RenderGlyphParams`, the sprite at the floored origin plus the raster offset).
    pub fn paint_glyph(
        &mut self,
        origin: Point<Pixels>,
        font_id: FontId,
        glyph_id: GlyphId,
        font_size: Pixels,
        color: Hsla,
    ) -> anyhow::Result<()> {
        let scale_factor = self.scale_factor;
        let glyph_origin = origin.scale(scale_factor);
        let subpixel_variant = Point {
            x: (glyph_origin.x.0.fract() * SUBPIXEL_VARIANTS_X as f32).floor() as u8,
            y: (glyph_origin.y.0.fract() * SUBPIXEL_VARIANTS_Y as f32).floor() as u8,
        };
        let params = RenderGlyphParams {
            font_id,
            glyph_id,
            font_size,
            subpixel_variant,
            scale_factor,
            is_emoji: false,
        };
        let Some((raster_bounds, mask)) =
            self.glyph_mask(&params, |size, bytes| Arc::new(AlphaMask::new(size, bytes)).into())?
        else {
            return Ok(());
        };
        let Some(mask) = mask.alpha().cloned() else { return Ok(()) };
        let bounds = Bounds {
            origin: glyph_origin.map(|px| px.floor()) + raster_bounds.origin.map(Into::into),
            size: mask.size.map(Into::into),
        };
        let clip = self.scaled_clip();
        if bounds.intersect(&clip).is_empty() {
            return Ok(());
        }
        let color = color.opacity(self.element_opacity);
        self.next_frame.scene.push_mono_sprite(MonoSprite {
            bounds,
            clip,
            color,
            mask,
            kind: SpriteKind::Glyph,
        });
        Ok(())
    }

    /// Paints a colour emoji glyph (no subpixel variants; a premultiplied BGRA mask).
    pub fn paint_emoji(
        &mut self,
        origin: Point<Pixels>,
        font_id: FontId,
        glyph_id: GlyphId,
        font_size: Pixels,
    ) -> anyhow::Result<()> {
        let scale_factor = self.scale_factor;
        let glyph_origin = origin.scale(scale_factor);
        let params = RenderGlyphParams {
            font_id,
            glyph_id,
            font_size,
            subpixel_variant: Point::default(),
            scale_factor,
            is_emoji: true,
        };
        let Some((raster_bounds, mask)) =
            self.glyph_mask(&params, |size, bytes| Arc::new(ColorMask::new(size, bytes)).into())?
        else {
            return Ok(());
        };
        let Some(mask) = mask.color().cloned() else { return Ok(()) };
        let bounds = Bounds {
            origin: glyph_origin.map(|px| px.floor()) + raster_bounds.origin.map(Into::into),
            size: mask.size.map(Into::into),
        };
        let clip = self.scaled_clip();
        if bounds.intersect(&clip).is_empty() {
            return Ok(());
        }
        let opacity = self.element_opacity;
        self.next_frame.scene.push_color_sprite(ColorSprite { bounds, clip, mask, opacity });
        Ok(())
    }

    /// The raster bounds and the cached mask of a glyph; `None` when it has no pixels.
    fn glyph_mask(
        &mut self,
        params: &RenderGlyphParams,
        make: impl FnOnce(Size<crate::DevicePixels>, Vec<u8>) -> Mask,
    ) -> anyhow::Result<Option<(Bounds<crate::DevicePixels>, Mask)>> {
        let text_system = &self.text_system;
        let raster_bounds = text_system.raster_bounds(params)?;
        if raster_bounds.size.width.0 == 0 || raster_bounds.size.height.0 == 0 {
            return Ok(None);
        }
        let mask = self.glyph_masks.get_or_insert_with(params, || {
            let (size, bytes) = text_system.rasterize_glyph(params)?;
            Ok(Some(make(size, bytes)))
        })?;
        Ok(mask.map(|mask| (raster_bounds, mask)))
    }

    /// Paints box shadows under a rounded rectangle.
    pub fn paint_shadows(
        &mut self,
        bounds: Bounds<Pixels>,
        corner_radii: Corners<Pixels>,
        shadows: &[BoxShadow],
    ) {
        let cx = self.style_paint();
        crate::paint_shadows(&mut self.next_frame.scene, &cx, bounds, corner_radii, shadows);
    }

    /// Paints an underline (straight, or wavy and three times as tall), as GPUI.
    pub fn paint_underline(&mut self, origin: Point<Pixels>, width: Pixels, style: &UnderlineStyle) {
        let height = if style.wavy { style.thickness * 3. } else { style.thickness };
        self.push_underline(
            origin,
            size(width, height),
            style.color.unwrap_or_default(),
            style.thickness,
            style.wavy,
        );
    }

    /// Paints a strikethrough.
    pub fn paint_strikethrough(&mut self, origin: Point<Pixels>, width: Pixels, style: &StrikethroughStyle) {
        self.push_underline(
            origin,
            size(width, style.thickness),
            style.color.unwrap_or_default(),
            style.thickness,
            false,
        );
    }

    fn push_underline(
        &mut self,
        origin: Point<Pixels>,
        line: Size<Pixels>,
        color: Hsla,
        thickness: Pixels,
        wavy: bool,
    ) {
        let scale_factor = self.scale_factor;
        let bounds = Bounds { origin, size: line }.scale(scale_factor);
        let clip = self.scaled_clip();
        if bounds.intersect(&clip).is_empty() {
            return;
        }
        let color = color.opacity(self.element_opacity);
        let thickness = thickness.scale(scale_factor);
        self.next_frame.scene.push_underline(Underline { bounds, clip, color, thickness, wavy });
    }

    /// Runs `f` as one layer. GPUI batches primitives by layer for the GPU; the CPU renderer has nothing to
    /// batch, so this only runs `f`.
    pub fn paint_layer<R>(&mut self, _bounds: Bounds<Pixels>, f: impl FnOnce(&mut Self) -> R) -> R {
        f(self)
    }
}
