// Adapted from GPUI (Apache-2.0): crates/gpui/src/assets.rs (AssetSource)
//! `AssetSource`: where the app's bundled files (SVG icons) come from.

use crate::SharedString;
use std::borrow::Cow;

/// A source of assets for the app.
pub trait AssetSource: 'static + Send + Sync {
    /// Loads the asset at `path`; `Ok(None)` when there is none.
    fn load(&self, path: &str) -> anyhow::Result<Option<Cow<'static, [u8]>>>;

    /// Lists the assets under `path`.
    fn list(&self, path: &str) -> anyhow::Result<Vec<SharedString>>;
}

/// No assets.
impl AssetSource for () {
    fn load(&self, _path: &str) -> anyhow::Result<Option<Cow<'static, [u8]>>> {
        Ok(None)
    }

    fn list(&self, _path: &str) -> anyhow::Result<Vec<SharedString>> {
        Ok(vec![])
    }
}
