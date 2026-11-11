//! Skeleton stand-ins for names the core uses that other packages own and have not landed yet. Each is the
//! smallest shape that lets the core compile; nothing here is the design of those types (`API.md` lists
//! what the app needs from each). Owner: none. **Each package deletes its block when its real types land**
//! (and the `pub use pending::*` line in `lib.rs` goes with the last block).
#![allow(dead_code)] // skeleton

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
