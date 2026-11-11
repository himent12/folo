//! Skeleton stand-ins for names the core uses that other packages own and have not landed yet. Each is the
//! smallest shape that lets the core compile; nothing here is the design of those types (`API.md` lists
//! what the app needs from each). Owner: none. **Each package deletes its block when its real types land**
//! (and the `pub use pending::*` line in `lib.rs` goes with the last block).
#![allow(dead_code)] // skeleton

// ---- windows package (platform/windows) ----

/// Stand-in for the Win32 backend's constructor, which `Application::new` calls on Windows with the
/// DirectWrite text system (`app/mod.rs`, `default_platform`). The windows package replaces the body
/// with its platform.
#[cfg(windows)]
pub mod platform_pending {
    /// The Win32 platform.
    pub fn windows_platform(
        _text_system: std::sync::Arc<dyn crate::PlatformTextSystem>,
    ) -> anyhow::Result<std::rc::Rc<dyn crate::Platform>> {
        anyhow::bail!(
            "the Win32 backend has not landed yet (run headless with `--shot`, or `FOLO_UI_PLATFORM=winit` \
             with the `winit` feature)"
        )
    }
}

// ---- raster package (raster/) ----

/// Stand-ins for the raster package's types (`raster::Renderer`, `raster::MaskCache`) live in
/// `raster/mod.rs` once it lands; until then the window refers to these.
pub mod raster_pending {
    use std::{collections::HashMap, hash::Hash, sync::Arc};

    use crate::{AlphaMask, Bounds, ColorMask, DevicePixels, Framebuffer, GlyphBlend, Scene};

    /// Stand-in: `raster::Renderer` (same signatures as `desktop/fw-raster`; draws nothing).
    #[derive(Default)]
    pub struct Renderer {}

    impl Renderer {
        /// A renderer.
        pub fn new() -> Self {
            Renderer {}
        }

        /// Makes `frame` show `scene`; the stand-in reports the whole frame as damaged.
        pub fn render(
            &mut self,
            _scene: &Scene,
            _glyph: GlyphBlend,
            frame: &mut Framebuffer,
        ) -> Vec<Bounds<DevicePixels>> {
            vec![Bounds::new(
                Default::default(),
                crate::size(DevicePixels(frame.width as i32), DevicePixels(frame.height as i32)),
            )]
        }

        /// The next frame repaints everything.
        pub fn invalidate(&mut self) {}
    }

    /// Stand-in: `raster::Mask`.
    #[derive(Clone, Debug)]
    pub enum Mask {
        /// A monochrome mask.
        Alpha(Arc<AlphaMask>),
        /// A colour mask.
        Color(Arc<ColorMask>),
    }

    impl Mask {
        /// The alpha mask, if this is one.
        pub fn alpha(&self) -> Option<&Arc<AlphaMask>> {
            match self {
                Mask::Alpha(m) => Some(m),
                Mask::Color(_) => None,
            }
        }

        /// The colour mask, if this is one.
        pub fn color(&self) -> Option<&Arc<ColorMask>> {
            match self {
                Mask::Color(m) => Some(m),
                Mask::Alpha(_) => None,
            }
        }
    }

    impl From<Arc<AlphaMask>> for Mask {
        fn from(m: Arc<AlphaMask>) -> Self {
            Mask::Alpha(m)
        }
    }

    impl From<Arc<ColorMask>> for Mask {
        fn from(m: Arc<ColorMask>) -> Self {
            Mask::Color(m)
        }
    }

    /// Stand-in: `raster::MaskCache<K>` (same signatures; unbounded).
    pub struct MaskCache<K> {
        masks: HashMap<K, Mask>,
    }

    impl<K: Hash + Eq + Clone> MaskCache<K> {
        /// An empty cache.
        pub fn new() -> Self {
            MaskCache { masks: HashMap::new() }
        }

        /// The mask for `key`; on a miss `make` renders it (`Ok(None)` is not cached).
        pub fn get_or_insert_with(
            &mut self,
            key: &K,
            make: impl FnOnce() -> anyhow::Result<Option<Mask>>,
        ) -> anyhow::Result<Option<Mask>> {
            if let Some(mask) = self.masks.get(key) {
                return Ok(Some(mask.clone()));
            }
            let Some(mask) = make()? else { return Ok(None) };
            self.masks.insert(key.clone(), mask.clone());
            Ok(Some(mask))
        }

        /// Starts a frame.
        pub fn begin_frame(&mut self) {}

        /// Ends a frame.
        pub fn end_frame(&mut self) {}

        /// How many masks are cached.
        pub fn len(&self) -> usize {
            self.masks.len()
        }

        /// Whether the cache is empty.
        pub fn is_empty(&self) -> bool {
            self.masks.is_empty()
        }
    }

    impl<K: Hash + Eq + Clone> Default for MaskCache<K> {
        fn default() -> Self {
            Self::new()
        }
    }
}
