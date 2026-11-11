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
    /// Stand-in: `raster::Renderer`.
    pub struct Renderer {}
    /// Stand-in: `raster::MaskCache`.
    pub struct MaskCache {}
}
