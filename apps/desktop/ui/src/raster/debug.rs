//! PNG files of framebuffers, for tests and tools.

use crate::Framebuffer;
use anyhow::{Context, Result};
use std::path::Path;

/// Writes `frame` as an RGB PNG.
pub fn write_png(frame: &Framebuffer, path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    let mut bytes = Vec::with_capacity(frame.pixels.len() * 3);
    for &p in &frame.pixels {
        bytes.extend_from_slice(&[(p >> 16) as u8, (p >> 8) as u8, p as u8]);
    }
    let img = ::image::RgbImage::from_raw(frame.width, frame.height, bytes).context("framebuffer size")?;
    img.save_with_format(path, ::image::ImageFormat::Png).with_context(|| format!("write {}", path.display()))
}

/// Reads a PNG (any colour type) into a framebuffer; alpha is dropped.
pub fn read_png(path: impl AsRef<Path>) -> Result<Framebuffer> {
    let path = path.as_ref();
    let img = ::image::open(path).with_context(|| format!("read {}", path.display()))?.to_rgb8();
    let (width, height) = img.dimensions();
    let pixels = img.pixels().map(|p| (p[0] as u32) << 16 | (p[1] as u32) << 8 | p[2] as u32).collect();
    Ok(Framebuffer { width, height, pixels })
}
