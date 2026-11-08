//! RGB images, decoding (with EXIF orientation), encoding and samples.

use crate::bbox::BBox;
use crate::error::{DataError, Result};
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

/// An 8-bit RGB image stored row-major, interleaved (`HWC`).
#[derive(Clone, PartialEq, Eq)]
pub struct Image {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// `width * height * 3` bytes, `R G B` per pixel, rows top to bottom.
    pub data: Vec<u8>,
}

impl std::fmt::Debug for Image {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Image({}x{})", self.width, self.height)
    }
}

impl Image {
    /// A `width x height` image filled with one colour.
    pub fn new(width: u32, height: u32, fill: [u8; 3]) -> Self {
        let n = width as usize * height as usize;
        let data = if fill[0] == fill[1] && fill[1] == fill[2] {
            vec![fill[0]; n * 3]
        } else {
            fill.iter().copied().cycle().take(n * 3).collect()
        };
        Image { width, height, data }
    }

    /// Wraps raw `HWC` RGB bytes; fails if the length does not match.
    pub fn from_raw(width: u32, height: u32, data: Vec<u8>) -> Result<Self> {
        if data.len() != width as usize * height as usize * 3 {
            return Err(DataError::Config(format!(
                "image buffer has {} bytes, expected {}x{}x3",
                data.len(),
                width,
                height
            )));
        }
        Ok(Image { width, height, data })
    }

    /// Bytes per row.
    #[inline]
    pub fn stride(&self) -> usize {
        self.width as usize * 3
    }

    /// The pixel at `(x, y)`.
    #[inline]
    pub fn pixel(&self, x: u32, y: u32) -> [u8; 3] {
        let i = (y as usize * self.width as usize + x as usize) * 3;
        [self.data[i], self.data[i + 1], self.data[i + 2]]
    }

    /// Sets the pixel at `(x, y)`.
    #[inline]
    pub fn set_pixel(&mut self, x: u32, y: u32, p: [u8; 3]) {
        let i = (y as usize * self.width as usize + x as usize) * 3;
        self.data[i..i + 3].copy_from_slice(&p);
    }

    /// Fills the pixel rectangle `[x0, x1) x [y0, y1)` (clipped to the image).
    pub fn fill_rect(&mut self, x0: i64, y0: i64, x1: i64, y1: i64, p: [u8; 3]) {
        let (w, h) = (self.width as i64, self.height as i64);
        let (x0, x1, y0, y1) = (x0.clamp(0, w), x1.clamp(0, w), y0.clamp(0, h), y1.clamp(0, h));
        let stride = self.stride();
        for y in y0..y1 {
            let row = &mut self.data[y as usize * stride..][x0 as usize * 3..x1 as usize * 3];
            for px in row.as_chunks_mut::<3>().0.iter_mut() {
                px.copy_from_slice(&p);
            }
        }
    }

    /// Applies an EXIF orientation (1–8) so the result is upright. Values outside 2–8 return a clone.
    pub fn oriented(&self, orientation: u8) -> Image {
        if !(2..=8).contains(&orientation) {
            return self.clone();
        }
        let (w, h) = (self.width as usize, self.height as usize);
        let (ow, oh) = if orientation >= 5 { (h, w) } else { (w, h) };
        let mut out = vec![0u8; ow * oh * 3];
        for y in 0..oh {
            for x in 0..ow {
                let (sx, sy) = match orientation {
                    2 => (w - 1 - x, y),
                    3 => (w - 1 - x, h - 1 - y),
                    4 => (x, h - 1 - y),
                    5 => (y, x),
                    6 => (y, h - 1 - x),
                    7 => (w - 1 - y, h - 1 - x),
                    _ => (w - 1 - y, x), // 8
                };
                let s = (sy * w + sx) * 3;
                let d = (y * ow + x) * 3;
                out[d..d + 3].copy_from_slice(&self.data[s..s + 3]);
            }
        }
        Image { width: ow as u32, height: oh as u32, data: out }
    }
}

/// An image with its boxes, as returned by [`crate::Dataset::get`].
#[derive(Clone, Debug, PartialEq)]
pub struct Sample {
    /// The decoded RGB image.
    pub image: Image,
    /// Boxes in normalised `cx, cy, w, h`.
    pub boxes: Vec<BBox>,
}

/// Image file extensions folo-data reads (lower case).
pub const IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "bmp", "webp"];

/// Whether `path` has a supported image extension.
pub fn is_image_path(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| IMAGE_EXTENSIONS.iter().any(|x| x.eq_ignore_ascii_case(e)))
}

/// Decodes an encoded image (JPEG, PNG, BMP, WebP) to upright RGB, honouring the JPEG EXIF orientation.
pub fn decode(bytes: &[u8]) -> Result<Image> {
    decode_with_path(bytes, Path::new(""))
}

pub(crate) fn decode_with_path(bytes: &[u8], path: &Path) -> Result<Image> {
    let dynimg = image::load_from_memory(bytes).map_err(|e| DataError::image(path, e))?;
    let rgb = dynimg.into_rgb8();
    let (width, height) = rgb.dimensions();
    if width == 0 || height == 0 {
        return Err(DataError::image(path, "image has zero size"));
    }
    let img = Image { width, height, data: rgb.into_raw() };
    match jpeg_exif_orientation(bytes) {
        Some(o) if o != 1 => Ok(img.oriented(o)),
        _ => Ok(img),
    }
}

/// Reads and decodes an image file.
pub fn read_image(path: &Path) -> Result<Image> {
    let bytes =
        crate::guard::check(path).and_then(|_| std::fs::read(path)).map_err(|e| DataError::io(path, e))?;
    decode_with_path(&bytes, path)
}

/// Size `(width, height)` of an image file after EXIF orientation, read from its header only.
pub fn probe_size(path: &Path) -> Result<(u32, u32)> {
    let file = crate::guard::check(path)
        .and_then(|_| std::fs::File::open(path))
        .map_err(|e| DataError::io(path, e))?;
    let mut reader = BufReader::new(file);
    let orientation = read_jpeg_exif_orientation(&mut reader).unwrap_or(1);
    std::io::Seek::rewind(&mut reader).map_err(|e| DataError::io(path, e))?;
    let file = reader.into_inner();
    let (w, h) = image::ImageReader::new(BufReader::new(file))
        .with_guessed_format()
        .map_err(|e| DataError::io(path, e))?
        .into_dimensions()
        .map_err(|e| DataError::image(path, e))?;
    if w == 0 || h == 0 {
        return Err(DataError::image(path, "image has zero size"));
    }
    Ok(if orientation >= 5 { (h, w) } else { (w, h) })
}

/// Streams the JPEG markers before the first scan (seeking over payloads) and returns the EXIF orientation, if any.
/// Unlike a fixed-size prefix this finds EXIF after large ICC or other APP segments.
fn read_jpeg_exif_orientation<R: Read + std::io::Seek>(r: &mut R) -> Option<u8> {
    let mut b2 = [0u8; 2];
    r.read_exact(&mut b2).ok()?;
    if b2 != [0xFF, 0xD8] {
        return None;
    }
    loop {
        r.read_exact(&mut b2).ok()?;
        if b2[0] != 0xFF {
            return None;
        }
        let mut marker = b2[1];
        while marker == 0xFF {
            let mut b = [0u8; 1];
            r.read_exact(&mut b).ok()?;
            marker = b[0];
        }
        if marker == 0xD8 || (0xD0..=0xD7).contains(&marker) || marker == 0x01 {
            continue;
        }
        if marker == 0xDA || marker == 0xD9 {
            return None;
        }
        r.read_exact(&mut b2).ok()?;
        let len = u16::from_be_bytes(b2) as usize;
        if len < 2 {
            return None;
        }
        if marker == 0xE1 {
            let mut seg = vec![0u8; len - 2];
            r.read_exact(&mut seg).ok()?;
            if seg.len() > 6
                && &seg[..6] == b"Exif\0\0"
                && let Some(o) = tiff_orientation(&seg[6..])
            {
                return Some(o);
            }
        } else {
            r.seek(std::io::SeekFrom::Current(len as i64 - 2)).ok()?;
        }
    }
}

/// The EXIF orientation tag (1–8) of a JPEG, if present. Only the bytes before the first scan are inspected, so a
/// prefix of the file that reaches the first scan is enough.
pub fn jpeg_exif_orientation(bytes: &[u8]) -> Option<u8> {
    if bytes.len() < 4 || bytes[0] != 0xFF || bytes[1] != 0xD8 {
        return None;
    }
    let mut i = 2;
    while i + 4 <= bytes.len() {
        if bytes[i] != 0xFF {
            return None;
        }
        let marker = bytes[i + 1];
        if marker == 0xFF {
            i += 1; // fill byte
            continue;
        }
        if marker == 0xD8 || (0xD0..=0xD7).contains(&marker) || marker == 0x01 {
            i += 2;
            continue;
        }
        if marker == 0xDA || marker == 0xD9 {
            return None; // start of scan / end of image: no EXIF before the pixel data
        }
        let len = u16::from_be_bytes([bytes[i + 2], bytes[i + 3]]) as usize;
        if len < 2 {
            return None;
        }
        let seg_end = (i + 2 + len).min(bytes.len());
        let seg = &bytes[i + 4..seg_end];
        if marker == 0xE1
            && seg.len() > 6
            && &seg[..6] == b"Exif\0\0"
            && let Some(o) = tiff_orientation(&seg[6..])
        {
            return Some(o);
        }
        i += 2 + len;
    }
    None
}

fn tiff_orientation(t: &[u8]) -> Option<u8> {
    let le = match t.get(..2)? {
        b"II" => true,
        b"MM" => false,
        _ => return None,
    };
    let u16_at = |o: usize| -> Option<u16> {
        let b = t.get(o..o.checked_add(2)?)?;
        Some(if le { u16::from_le_bytes([b[0], b[1]]) } else { u16::from_be_bytes([b[0], b[1]]) })
    };
    let u32_at = |o: usize| -> Option<u32> {
        let b = t.get(o..o.checked_add(4)?)?;
        let a = [b[0], b[1], b[2], b[3]];
        Some(if le { u32::from_le_bytes(a) } else { u32::from_be_bytes(a) })
    };
    if u16_at(2)? != 42 {
        return None;
    }
    let ifd = u32_at(4)? as usize;
    let count = u16_at(ifd)? as usize;
    for k in 0..count.min(512) {
        let e = ifd + 2 + k * 12;
        if u16_at(e)? == 0x0112 {
            let v = match u16_at(e + 2)? {
                3 => u16_at(e + 8)? as u32, // SHORT
                4 => u32_at(e + 8)?,        // LONG (non-standard but seen in the wild)
                _ => return None,
            };
            return (1..=8).contains(&v).then_some(v as u8);
        }
    }
    None
}

/// Output encoding for [`encode`] / [`write_image`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageFormat {
    /// Baseline JPEG with the given quality (1–100).
    Jpeg(u8),
    /// Lossless PNG.
    Png,
}

impl ImageFormat {
    /// File extension without the dot.
    pub fn extension(self) -> &'static str {
        match self {
            ImageFormat::Jpeg(_) => "jpg",
            ImageFormat::Png => "png",
        }
    }
}

/// Encodes an image to JPEG or PNG bytes.
pub fn encode(img: &Image, format: ImageFormat) -> Result<Vec<u8>> {
    use image::ImageEncoder;
    let mut out = Vec::new();
    let color = image::ExtendedColorType::Rgb8;
    let res = match format {
        ImageFormat::Jpeg(q) => image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, q.clamp(1, 100))
            .write_image(&img.data, img.width, img.height, color),
        ImageFormat::Png => {
            image::codecs::png::PngEncoder::new(&mut out).write_image(&img.data, img.width, img.height, color)
        }
    };
    res.map_err(|e| DataError::image(PathBuf::new(), e))?;
    Ok(out)
}

/// Encodes and writes an image file.
pub fn write_image(img: &Image, path: &Path, format: ImageFormat) -> Result<()> {
    let bytes = encode(img, format)?;
    std::fs::write(path, bytes).map_err(|e| DataError::io(path, e))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid() -> Image {
        let mut img = Image::new(3, 2, [0, 0, 0]);
        for y in 0..2 {
            for x in 0..3 {
                img.set_pixel(x, y, [x as u8 * 10, y as u8 * 10, 1]);
            }
        }
        img
    }

    #[test]
    fn orientations() {
        let img = grid();
        // 6: rotate 90 CW. The new top-left is the old bottom-left.
        let r = img.oriented(6);
        assert_eq!((r.width, r.height), (2, 3));
        assert_eq!(r.pixel(0, 0), img.pixel(0, 1));
        assert_eq!(r.pixel(1, 0), img.pixel(0, 0));
        // 8: rotate 90 CCW. The new top-left is the old top-right.
        assert_eq!(img.oriented(8).pixel(0, 0), img.pixel(2, 0));
        assert_eq!(img.oriented(3).pixel(0, 0), img.pixel(2, 1));
        assert_eq!(img.oriented(2).oriented(2), img);
        assert_eq!(img.oriented(6).oriented(8), img);
        assert_eq!(img.oriented(5).oriented(5), img);
        assert_eq!(img.oriented(7).oriented(7), img);
    }

    #[test]
    fn encode_decode_png_exact() {
        let img = grid();
        let png = encode(&img, ImageFormat::Png).unwrap();
        assert_eq!(decode(&png).unwrap(), img);
        assert!(decode(b"not an image").is_err());
        assert!(decode(&png[..png.len() / 2]).is_err());
    }
}
