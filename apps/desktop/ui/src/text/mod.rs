//! Text: the value types and `PlatformTextSystem` (`types.rs`), `TextSystem`, line layout, wrapping and
//! painting (the rest), and the DirectWrite and cosmic-text backends.

#[cfg(any(test, feature = "test-support"))]
mod fixed;
mod types;

#[cfg(windows)]
pub mod dwrite;
#[cfg(any(not(windows), feature = "portable-text"))]
pub mod portable;

#[cfg(any(test, feature = "test-support"))]
pub use fixed::*;
pub use types::*;
