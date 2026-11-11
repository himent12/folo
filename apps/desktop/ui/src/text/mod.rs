//! Text: the value types and `PlatformTextSystem` (`types.rs`), `TextSystem`, line layout, wrapping and
//! painting (the rest), and the DirectWrite and cosmic-text backends.

#[cfg(any(test, feature = "test-support"))]
mod fixed;
pub(crate) mod fx_hash;
mod layout_cache;
mod line;
mod line_layout;
mod line_wrapper;
mod paint;
mod system;
mod types;
mod window_text_system;

#[cfg(windows)]
pub mod dwrite;
#[cfg(any(not(windows), feature = "portable-text"))]
pub mod portable;

#[cfg(test)]
mod tests;

#[cfg(any(test, feature = "test-support"))]
pub use fixed::*;
pub use layout_cache::LineLayoutIndex;
pub use line::*;
pub use line_layout::*;
pub use line_wrapper::*;
pub use paint::*;
pub use system::*;
pub use types::*;
pub use window_text_system::*;
