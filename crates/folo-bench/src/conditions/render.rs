//! Rendering a variant from an image file: decode, apply the condition, encode the canonical JPEG.

use folo_api::BBox;

use crate::Error;

use super::Condition;
use super::encoding::{decode, encode_jpeg};
use super::transform::{apply, level};

/// A variant as it is served: the canonical JPEG and the true boxes in its coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct Rendered {
    /// The JPEG bytes ([`encode_jpeg`]).
    pub jpeg: Vec<u8>,
    /// Width in pixels (the original's, after EXIF orientation).
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// The true boxes under the condition ([`apply`]).
    pub boxes: Vec<BBox>,
}

/// An image file in, the variant out: [`decode`], [`apply`], [`encode_jpeg`]. `boxes` are the image's true boxes
/// in pixels of the upright image; `seed` is the variant's ([`super::variant_seed`]).
pub fn render(
    bytes: &[u8],
    condition: Condition,
    severity: u8,
    boxes: &[BBox],
    seed: u64,
) -> Result<Rendered, Error> {
    level(severity)?;
    let img = decode(bytes)?;
    let (img, boxes) = apply(condition, severity, &img, boxes, seed)?;
    Ok(Rendered { jpeg: encode_jpeg(&img)?, width: img.width(), height: img.height(), boxes })
}
