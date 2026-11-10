//! Variant seeds, decoding an image file, the canonical JPEG and rendering a variant from a file.

use folo_api::BBox;
use image::RgbImage;

use crate::Error;
use crate::pixels::mix;

use super::transform::{apply, level};
use super::{CANONICAL_QUALITY, Condition};

// ---------------------------------------------------------------------------------------------------------------
// Seeds and encoding
// ---------------------------------------------------------------------------------------------------------------

/// FNV-1a of a string: a stable 64-bit key.
fn key(s: &str) -> u64 {
    s.bytes().fold(0xCBF2_9CE4_8422_2325, |h, b| (h ^ b as u64).wrapping_mul(0x0100_0000_01B3))
}

/// The seed of one variant: a function of the benchmark's seed, a stable key of the image (its id within the
/// benchmark), the condition's id and the severity. Equal inputs give equal seeds on every platform and version
/// of this crate.
pub fn variant_seed(bench_seed: u64, image_key: &str, condition: Condition, severity: u8) -> u64 {
    mix(mix(mix(bench_seed, key(image_key)), key(condition.id())), severity as u64)
}

/// The canonical encoding of a variant: a baseline JPEG at quality [`CANONICAL_QUALITY`]. Fails for an empty
/// image or one with a side over 65,535 px.
pub fn encode_jpeg(image: &RgbImage) -> Result<Vec<u8>, Error> {
    encode_jpeg_quality(image, CANONICAL_QUALITY)
}

pub(super) fn encode_jpeg_quality(image: &RgbImage, quality: u8) -> Result<Vec<u8>, Error> {
    use image::ImageEncoder;
    if image.width() == 0 || image.height() == 0 {
        return Err(Error::Encode("the image is empty".into()));
    }
    let mut out = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, quality)
        .write_image(image.as_raw(), image.width(), image.height(), image::ExtendedColorType::Rgb8)
        .map_err(|e| Error::Encode(e.to_string()))?;
    Ok(out)
}

/// Decodes an image file (JPEG, PNG, BMP, WebP) to upright RGB exactly as datasets are read
/// ([`folo_data::decode`]: the JPEG EXIF orientation is applied), so the pixels line up with the truth boxes.
pub fn decode(bytes: &[u8]) -> Result<RgbImage, Error> {
    let img = folo_data::decode(bytes).map_err(|e| Error::Decode(e.to_string()))?;
    Ok(RgbImage::from_raw(img.width, img.height, img.data).expect("folo-data images are width x height x 3"))
}

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
/// in pixels of the upright image; `seed` is the variant's ([`variant_seed`]).
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
