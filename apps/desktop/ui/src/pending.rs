//! Skeleton stand-ins for names the core uses that other packages own and have not landed yet. Each is the
//! smallest shape that lets the core compile; nothing here is the design of those types (`API.md` lists
//! what the app needs from each). Owner: none. **Each package deletes its block when its real types land**
//! (and the `pub use pending::*` line in `lib.rs` goes with the last block).
#![allow(dead_code)] // skeleton

// ---- style package (style.rs, styled.rs, layout.rs, path_builder.rs, assets.rs) ----

/// Stand-in: the style package's `Style`.
#[derive(Clone, Debug, Default)]
pub struct Style {}

/// Stand-in: the style package's `StyleRefinement`.
#[derive(Clone, Debug, Default)]
pub struct StyleRefinement {}

/// Stand-in: the style package's `TextStyle`.
#[derive(Clone, Debug, Default)]
pub struct TextStyle {}

/// Stand-in: the style package's `TextStyleRefinement`.
#[derive(Clone, Debug, Default)]
pub struct TextStyleRefinement {}

/// Stand-in: the style package's `BoxShadow`.
#[derive(Clone, Debug, Default)]
pub struct BoxShadow {}

/// Stand-in: the style package's `AvailableSpace` (GPUI's taffy.rs).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum AvailableSpace {
    /// A definite size.
    Definite(crate::Pixels),
    /// Min-content.
    #[default]
    MinContent,
    /// Max-content.
    MaxContent,
}

/// Stand-in: what `PathBuilder::build()` returns (logical px), which `Window::paint_path` takes.
#[derive(Clone, Debug)]
pub struct BuiltPath {}

/// Stand-in: the style package's `Styled` (the real one adds the fluent methods).
pub trait Styled: Sized {
    /// The element's style refinement.
    fn style(&mut self) -> &mut StyleRefinement;
}

/// Stand-in: the style package's `AssetSource` (GPUI 0.2.2's trait; the app implements it in `main.rs`).
pub trait AssetSource: 'static {
    /// Loads an asset.
    fn load(&self, path: &str) -> anyhow::Result<Option<std::borrow::Cow<'static, [u8]>>>;
    /// Lists assets under `path`.
    fn list(&self, path: &str) -> anyhow::Result<Vec<crate::SharedString>>;
}

impl AssetSource for () {
    fn load(&self, _path: &str) -> anyhow::Result<Option<std::borrow::Cow<'static, [u8]>>> {
        Ok(None)
    }

    fn list(&self, _path: &str) -> anyhow::Result<Vec<crate::SharedString>> {
        Ok(Vec::new())
    }
}

// ---- text package (text/) ----

/// Stand-in: the text package's app-wide `TextSystem`.
pub struct TextSystem {}

/// Stand-in: the text package's per-window `WindowTextSystem`.
pub struct WindowTextSystem {}

/// Stand-in: the text package's `WrappedLine`.
pub struct WrappedLine {}

// ---- raster package (raster/) ----

/// Stand-ins for the raster package's types (`raster::Renderer`, `raster::MaskCache`) live in
/// `raster/mod.rs` once it lands; until then the window refers to these.
pub mod raster_pending {
    /// Stand-in: `raster::Renderer`.
    pub struct Renderer {}
    /// Stand-in: `raster::MaskCache`.
    pub struct MaskCache {}
}
