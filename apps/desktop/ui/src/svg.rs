// Adapted from GPUI (Apache-2.0): crates/gpui/src/svg_renderer.rs
//! SVG icons to alpha masks, as GPUI renders them: resvg at the requested width (twice the device size,
//! [`SMOOTH_SVG_SCALE_FACTOR`]), alpha taken from the pixmap. Parsed trees are cached per path.

use crate::{AlphaMask, AssetSource, Bounds, DevicePixels, IsZero, ScaledPixels, SharedString, Size};
use parking_lot::Mutex;
use resvg::tiny_skia::{Pixmap, Transform};
use std::borrow::Borrow;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// SVGs are rendered at twice their device size and drawn scaled down, for smoother edges.
pub const SMOOTH_SVG_SCALE_FACTOR: f32 = 2.;

/// What to render: an asset path at a mask size in device pixels.
#[derive(Clone, Debug, Eq)]
pub struct RenderSvgParams {
    /// The asset path.
    pub path: SharedString,
    /// The mask size (already multiplied by [`SMOOTH_SVG_SCALE_FACTOR`]).
    pub size: Size<DevicePixels>,
}

/// Borrowed view of [`RenderSvgParams`] used for zero-clone cache lookups.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct RenderSvgParamsRef<'a> {
    pub path: &'a str,
    pub size: Size<DevicePixels>,
}

/// Trait object bridge allowing `MaskCache<RenderSvgParams>` lookups with [`RenderSvgParamsRef`].
pub(crate) trait AsRenderSvgKeyRef {
    fn as_render_svg_key_ref(&self) -> RenderSvgParamsRef<'_>;
}

impl AsRenderSvgKeyRef for RenderSvgParams {
    fn as_render_svg_key_ref(&self) -> RenderSvgParamsRef<'_> {
        RenderSvgParamsRef { path: self.path.as_str(), size: self.size }
    }
}

impl AsRenderSvgKeyRef for RenderSvgParamsRef<'_> {
    fn as_render_svg_key_ref(&self) -> RenderSvgParamsRef<'_> {
        *self
    }
}

impl PartialEq for dyn AsRenderSvgKeyRef + '_ {
    fn eq(&self, other: &Self) -> bool {
        self.as_render_svg_key_ref() == other.as_render_svg_key_ref()
    }
}

impl Eq for dyn AsRenderSvgKeyRef + '_ {}

impl Hash for dyn AsRenderSvgKeyRef + '_ {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_render_svg_key_ref().hash(state);
    }
}

impl PartialEq for RenderSvgParams {
    fn eq(&self, other: &Self) -> bool {
        self.as_render_svg_key_ref() == other.as_render_svg_key_ref()
    }
}

impl Hash for RenderSvgParams {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_render_svg_key_ref().hash(state);
    }
}

impl<'a> Borrow<dyn AsRenderSvgKeyRef + 'a> for RenderSvgParams {
    fn borrow(&self) -> &(dyn AsRenderSvgKeyRef + 'a) {
        self as &dyn AsRenderSvgKeyRef
    }
}

impl RenderSvgParams {
    /// GPUI's mask size for an SVG drawn into `bounds` (device pixels): each side times
    /// [`SMOOTH_SVG_SCALE_FACTOR`], rounded up.
    pub(crate) fn size_for_bounds(bounds: &Bounds<ScaledPixels>) -> Size<DevicePixels> {
        bounds.size.map(|pixels| DevicePixels((pixels.0 * SMOOTH_SVG_SCALE_FACTOR).ceil() as i32))
    }

    /// GPUI's params for an SVG drawn into `bounds` (device pixels): each side times
    /// [`SMOOTH_SVG_SCALE_FACTOR`], rounded up.
    pub fn for_bounds(path: SharedString, bounds: &Bounds<ScaledPixels>) -> Self {
        Self { path, size: Self::size_for_bounds(bounds) }
    }
}

/// The size to render an SVG at.
pub enum SvgSize {
    /// Scaled so its width is this many pixels (the height follows the aspect ratio).
    Size(Size<DevicePixels>),
    /// Scaled by this factor.
    ScaleFactor(f32),
}

/// Renders SVG assets to alpha masks.
#[derive(Clone)]
pub struct SvgRenderer {
    asset_source: Arc<dyn AssetSource>,
    usvg_options: Arc<usvg::Options<'static>>,
    trees: Arc<Mutex<HashMap<SharedString, Arc<usvg::Tree>>>>,
}

impl SvgRenderer {
    /// A renderer loading SVGs from `asset_source`.
    pub fn new(asset_source: Arc<dyn AssetSource>) -> Self {
        Self { asset_source, usvg_options: Arc::new(usvg::Options::default()), trees: Arc::default() }
    }

    /// Renders `params.path` at `params.size` into an alpha mask. `Ok(None)` when the asset does not exist.
    pub fn render(&self, params: &RenderSvgParams) -> anyhow::Result<Option<AlphaMask>> {
        self.render_ref(&params.path, params.size)
    }

    /// Renders `path` at `size` into an alpha mask without cloning `path` on parsed-tree cache hits.
    pub(crate) fn render_ref(
        &self,
        path: &SharedString,
        size: Size<DevicePixels>,
    ) -> anyhow::Result<Option<AlphaMask>> {
        anyhow::ensure!(!size.is_zero(), "can't render at a zero size");
        let Some(tree) = self.tree(path)? else {
            return Ok(None);
        };
        let pixmap = render_tree(&tree, SvgSize::Size(size))?;
        let size = Size::new(DevicePixels(pixmap.width() as i32), DevicePixels(pixmap.height() as i32));
        let alpha_mask: Vec<u8> = pixmap.pixels().iter().map(|p| p.alpha()).collect();
        Ok(Some(AlphaMask::new(size, alpha_mask)))
    }

    /// Renders SVG `bytes` to a pixmap (premultiplied RGBA), as GPUI's `render_pixmap`.
    pub fn render_pixmap(&self, bytes: &[u8], size: SvgSize) -> Result<Pixmap, usvg::Error> {
        let tree = usvg::Tree::from_data(bytes, &self.usvg_options)?;
        render_tree(&tree, size)
    }

    /// The parsed tree of `path`, from the cache or loaded and parsed now.
    fn tree(&self, path: &SharedString) -> anyhow::Result<Option<Arc<usvg::Tree>>> {
        if let Some(tree) = self.trees.lock().get(path.as_str()) {
            return Ok(Some(tree.clone()));
        }
        let Some(bytes) = self.asset_source.load(path.as_str())? else {
            return Ok(None);
        };
        let tree = Arc::new(usvg::Tree::from_data(&bytes, &self.usvg_options)?);
        self.trees.lock().insert(path.clone(), tree.clone());
        Ok(Some(tree))
    }
}

/// Renders `tree` at `size`: the scale is the requested width over the SVG's, and the pixmap is the SVG's
/// size times that scale, truncated (GPUI's arithmetic, so masks have the same dimensions).
fn render_tree(tree: &usvg::Tree, size: SvgSize) -> Result<Pixmap, usvg::Error> {
    let svg_size = tree.size();
    let scale = match size {
        SvgSize::Size(size) => size.width.0 as f32 / svg_size.width(),
        SvgSize::ScaleFactor(scale) => scale,
    };
    let mut pixmap = Pixmap::new((svg_size.width() * scale) as u32, (svg_size.height() * scale) as u32)
        .ok_or(usvg::Error::InvalidSize)?;
    resvg::render(tree, Transform::from_scale(scale, scale), &mut pixmap.as_mut());
    Ok(pixmap)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raster::{Mask, MaskCache};

    #[test]
    fn borrowed_svg_key_matches_owned_and_avoids_cloning_on_hit() {
        let mut cache: MaskCache<RenderSvgParams> = MaskCache::new();
        let raw: Arc<str> = Arc::from("icons/dynamic-check.svg");
        let path = SharedString::new(Arc::clone(&raw));
        let size = Size::new(DevicePixels(32), DevicePixels(32));
        let key_ref = RenderSvgParamsRef { path: path.as_str(), size };

        let mut make_key_calls = 0usize;
        let mut make_val_calls = 0usize;
        for _ in 0..3 {
            let mask = cache
                .get_or_insert_with_by(
                    &key_ref as &dyn AsRenderSvgKeyRef,
                    || {
                        make_key_calls += 1;
                        RenderSvgParams { path: path.clone(), size }
                    },
                    || {
                        make_val_calls += 1;
                        Ok(Some(Mask::Alpha(Arc::new(AlphaMask::new(size, vec![1u8; 32 * 32])))))
                    },
                )
                .unwrap()
                .unwrap();
            assert_eq!(mask.into_alpha().unwrap().size, size);
        }
        assert_eq!(make_key_calls, 1, "owned key should only be constructed on miss");
        assert_eq!(make_val_calls, 1, "mask should only be rendered on miss");
        // Owned by: `raw`, `path`, and the single cache entry.
        assert_eq!(Arc::strong_count(&raw), 3);
    }
}
