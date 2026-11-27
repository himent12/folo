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
        let total = (width as usize)
            .checked_mul(height as usize)
            .and_then(|n| n.checked_mul(3))
            .expect("image dimensions overflow");
        let data = if fill[0] == fill[1] && fill[1] == fill[2] {
            vec![fill[0]; total]
        } else {
            let mut buf = vec![0u8; total];
            fill_rgb_pattern(&mut buf, fill);
            buf
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
        if x0 >= x1 || y0 >= y1 {
            return;
        }
        let stride = self.stride();
        let x0b = x0 as usize * 3;
        let x1b = x1 as usize * 3;
        let is_grey = p[0] == p[1] && p[1] == p[2];
        for y in y0..y1 {
            let row = &mut self.data[y as usize * stride..][x0b..x1b];
            if is_grey {
                row.fill(p[0]);
            } else {
                fill_rgb_pattern(row, p);
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
        if w == 0 || h == 0 {
            return Image { width: ow as u32, height: oh as u32, data: Vec::new() };
        }
        crate::simd::assert_image_data(self);
        let mut out = crate::buffers::scratch(ow * oh * 3);
        let sstride = w * 3;
        let ostride = ow * 3;
        match orientation {
            2 => crate::par::for_rows(&mut out, ostride, |y, dst_row| {
                crate::simd::planar::copy_flip_lr_row(&self.data[y * sstride..(y + 1) * sstride], dst_row);
            }),
            3 => crate::par::for_rows(&mut out, ostride, |y, dst_row| {
                let sy = h - 1 - y;
                crate::simd::planar::copy_flip_lr_row(&self.data[sy * sstride..(sy + 1) * sstride], dst_row);
            }),
            4 => crate::par::for_rows(&mut out, ostride, |y, dst_row| {
                let sy = h - 1 - y;
                dst_row.copy_from_slice(&self.data[sy * sstride..(sy + 1) * sstride]);
            }),
            _ => {
                // Orientations 5–8 transpose `(w, h) -> (h, w)`. 32x32 cache-blocked tiles keep both
                // source and destination blocks (~3 KiB each) resident in L1d cache instead of striding
                // across cache lines per pixel.
                const TILE: usize = 32;
                let src = &self.data;
                crate::par::for_bands(&mut out, ostride, |y0, band| {
                    let band_h = band.len() / ostride;
                    for ty in (0..band_h).step_by(TILE) {
                        let y_end = (ty + TILE).min(band_h);
                        for tx in (0..ow).step_by(TILE) {
                            let x_end = (tx + TILE).min(ow);
                            match orientation {
                                5 => {
                                    for ly in ty..y_end {
                                        let sx = y0 + ly;
                                        let drow = &mut band[ly * ostride..][..ostride];
                                        for x in tx..x_end {
                                            let s = (x * w + sx) * 3;
                                            let d = x * 3;
                                            drow[d..d + 3].copy_from_slice(&src[s..s + 3]);
                                        }
                                    }
                                }
                                6 => {
                                    for ly in ty..y_end {
                                        let sx = y0 + ly;
                                        let drow = &mut band[ly * ostride..][..ostride];
                                        for x in tx..x_end {
                                            let s = ((h - 1 - x) * w + sx) * 3;
                                            let d = x * 3;
                                            drow[d..d + 3].copy_from_slice(&src[s..s + 3]);
                                        }
                                    }
                                }
                                7 => {
                                    for ly in ty..y_end {
                                        let sx = w - 1 - (y0 + ly);
                                        let drow = &mut band[ly * ostride..][..ostride];
                                        for x in tx..x_end {
                                            let s = ((h - 1 - x) * w + sx) * 3;
                                            let d = x * 3;
                                            drow[d..d + 3].copy_from_slice(&src[s..s + 3]);
                                        }
                                    }
                                }
                                _ => {
                                    // 8
                                    for ly in ty..y_end {
                                        let sx = w - 1 - (y0 + ly);
                                        let drow = &mut band[ly * ostride..][..ostride];
                                        for x in tx..x_end {
                                            let s = (x * w + sx) * 3;
                                            let d = x * 3;
                                            drow[d..d + 3].copy_from_slice(&src[s..s + 3]);
                                        }
                                    }
                                }
                            }
                        }
                    }
                });
            }
        }
        Image { width: ow as u32, height: oh as u32, data: out }
    }
}

/// Fills an interleaved RGB slice with one colour using doubling `copy_within` (`memcpy`) for non-grey fills.
#[inline]
fn fill_rgb_pattern(buf: &mut [u8], fill: [u8; 3]) {
    let n = buf.len();
    if n < 3 {
        return;
    }
    if fill[0] == fill[1] && fill[1] == fill[2] {
        buf.fill(fill[0]);
        return;
    }
    let seed_len = n.min(48);
    for px in buf[..seed_len].as_chunks_mut::<3>().0.iter_mut() {
        *px = fill;
    }
    let mut filled = seed_len - (seed_len % 3);
    while filled < n {
        let chunk = filled.min(n - filled);
        buf.copy_within(0..chunk, filled);
        filled += chunk;
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
    let err = |e: image::ImageError| DataError::image(path, e.to_string().trim());
    let img = match decode_jpeg(bytes).or_else(|| decode_png(bytes)) {
        Some(img) => img.map_err(err)?,
        None => {
            let rgb = image::load_from_memory(bytes).map_err(err)?.into_rgb8();
            let (width, height) = rgb.dimensions();
            Image { width, height, data: rgb.into_raw() }
        }
    };
    if img.width == 0 || img.height == 0 {
        return Err(DataError::image(path, "image has zero size"));
    }
    match jpeg_exif_orientation(bytes) {
        Some(o) if o != 1 => {
            let upright = img.oriented(o);
            crate::buffers::recycle(img.data);
            Ok(upright)
        }
        _ => Ok(img),
    }
}

/// Fallback conversion through `DynamicImage::from_decoder` for less-common color types.
fn decode_dynamic(dec: impl image::ImageDecoder) -> image::ImageResult<Image> {
    image::DynamicImage::from_decoder(dec).map(|d| {
        let rgb = d.into_rgb8();
        let (width, height) = rgb.dimensions();
        Image { width, height, data: rgb.into_raw() }
    })
}

/// `image::load_from_memory(bytes)?.into_rgb8()` for an RGB or greyscale JPEG (the same decoder, limits, bytes and
/// errors), decoding into a buffer from [`crate::buffers`]; `None` for anything else.
fn decode_jpeg(bytes: &[u8]) -> Option<image::ImageResult<Image>> {
    use image::{ColorType, ImageDecoder};
    if image::guess_format(bytes).ok()? != image::ImageFormat::Jpeg {
        return None;
    }
    let open = || {
        let mut dec = image::codecs::jpeg::JpegDecoder::new(std::io::Cursor::new(bytes))?;
        // What `ImageReader::decode` checks and sets.
        let mut limits = image::Limits::default();
        limits.reserve(dec.total_bytes())?;
        dec.set_limits(limits)?;
        Ok(dec)
    };
    let dec = match open() {
        Ok(d) => d,
        Err(e) => return Some(Err(e)),
    };
    let (width, height) = dec.dimensions();
    let Some(n) = (width as usize).checked_mul(height as usize) else {
        return Some(decode_dynamic(dec));
    };
    let Some(rgb_bytes) = n.checked_mul(3) else {
        return Some(decode_dynamic(dec));
    };
    Some(match dec.color_type() {
        ColorType::Rgb8 => {
            let mut data = crate::buffers::zeroed(rgb_bytes);
            dec.read_image(&mut data).map(|()| Image { width, height, data })
        }
        ColorType::L8 => {
            let mut luma = crate::buffers::zeroed(n);
            dec.read_image(&mut luma).map(|()| {
                // `into_rgb8` of a greyscale image: the level in all three channels.
                let mut data = crate::buffers::scratch(rgb_bytes);
                crate::simd::planar::luma_to_rgb(&luma, &mut data);
                crate::buffers::recycle(luma);
                Image { width, height, data }
            })
        }
        _ => decode_dynamic(dec),
    })
}

/// `image::load_from_memory(bytes)?.into_rgb8()` for an RGB, RGBA or greyscale PNG (the same decoder, limits, bytes
/// and errors), decoding into pooled buffers from [`crate::buffers`] with AVX2 `RGBA`/`L8` → `RGB` conversion; `None`
/// for anything else.
fn decode_png(bytes: &[u8]) -> Option<image::ImageResult<Image>> {
    use image::{ColorType, ImageDecoder};
    if image::guess_format(bytes).ok()? != image::ImageFormat::Png {
        return None;
    }
    let open = || {
        let mut limits = image::Limits::default();
        let mut dec =
            image::codecs::png::PngDecoder::with_limits(std::io::Cursor::new(bytes), limits.clone())?;
        limits.reserve(dec.total_bytes())?;
        dec.set_limits(limits)?;
        Ok(dec)
    };
    let dec = match open() {
        Ok(d) => d,
        Err(e) => return Some(Err(e)),
    };
    let (width, height) = dec.dimensions();
    let Some(n) = (width as usize).checked_mul(height as usize) else {
        return Some(decode_dynamic(dec));
    };
    let Some(rgb_bytes) = n.checked_mul(3) else {
        return Some(decode_dynamic(dec));
    };
    Some(match dec.color_type() {
        ColorType::Rgb8 => {
            let mut data = crate::buffers::zeroed(rgb_bytes);
            dec.read_image(&mut data).map(|()| Image { width, height, data })
        }
        ColorType::Rgba8 => match n.checked_mul(4) {
            Some(rgba_bytes) => {
                let mut rgba = crate::buffers::zeroed(rgba_bytes);
                dec.read_image(&mut rgba).map(|()| {
                    let mut data = crate::buffers::scratch(rgb_bytes);
                    crate::simd::planar::rgba_to_rgb(&rgba, &mut data);
                    crate::buffers::recycle(rgba);
                    Image { width, height, data }
                })
            }
            None => decode_dynamic(dec),
        },
        ColorType::L8 => {
            let mut luma = crate::buffers::zeroed(n);
            dec.read_image(&mut luma).map(|()| {
                let mut data = crate::buffers::scratch(rgb_bytes);
                crate::simd::planar::luma_to_rgb(&luma, &mut data);
                crate::buffers::recycle(luma);
                Image { width, height, data }
            })
        }
        _ => decode_dynamic(dec),
    })
}

/// Reads and decodes an image file.
pub fn read_image(path: &Path) -> Result<Image> {
    crate::buffers::read_file_bytes(path, |bytes| decode_with_path(bytes, path))
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
    let (w, h) = header_reader(BufReader::new(file))
        .with_guessed_format()
        .map_err(|e| DataError::io(path, e))?
        .into_dimensions()
        .map_err(|e| DataError::image(path, e))?;
    if w == 0 || h == 0 {
        return Err(DataError::image(path, "image has zero size"));
    }
    Ok(if orientation >= 5 { (h, w) } else { (w, h) })
}

/// A buffered, seekable byte source `image` can read a header from.
pub(crate) trait HeaderSource: std::io::BufRead + std::io::Seek {}
impl<T: std::io::BufRead + std::io::Seek> HeaderSource for T {}

/// `image`'s reader over `src`, behind one reader type for files and memory alike. `image` compiles its decoders
/// once for every reader type it is handed (a second one was a tenth of this crate's code to optimise); a header
/// is a handful of reads, so the indirection costs nothing measurable.
pub(crate) fn header_reader<'a>(
    src: impl HeaderSource + 'a,
) -> image::ImageReader<Box<dyn HeaderSource + 'a>> {
    image::ImageReader::new(Box::new(src))
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

/// True for a JPEG whose data ends before its end-of-image marker: the file was cut off mid-scan (a partial
/// download or copy). Decoders accept such files and fill the missing rows with grey, so a full decode alone does
/// not notice. The markers before the first scan are walked by their lengths (an EXIF thumbnail carries its own
/// EOI), then the rest must contain `FF D9`, which never occurs inside entropy-coded data (`FF` is stuffed as
/// `FF 00` there, restart markers are `FF D0`..`FF D7`). Non-JPEG bytes are never truncated by this test.
pub fn jpeg_truncated(bytes: &[u8]) -> bool {
    if bytes.len() < 4 || bytes[0] != 0xFF || bytes[1] != 0xD8 {
        return false;
    }
    let mut i = 2;
    loop {
        if i + 2 > bytes.len() {
            return true; // the headers themselves are cut off
        }
        if bytes[i] != 0xFF {
            return false; // not a marker: leave it to the decoder
        }
        let marker = bytes[i + 1];
        if marker == 0xFF {
            i += 1;
            continue;
        }
        if marker == 0xD8 || (0xD0..=0xD7).contains(&marker) || marker == 0x01 {
            i += 2;
            continue;
        }
        if marker == 0xD9 {
            return false; // EOI before any scan
        }
        if i + 4 > bytes.len() {
            return true;
        }
        let len = u16::from_be_bytes([bytes[i + 2], bytes[i + 3]]) as usize;
        if marker == 0xDA {
            let scan = (i + 2 + len).min(bytes.len());
            return !bytes[scan..].windows(2).any(|w| w == [0xFF, 0xD9]);
        }
        if len < 2 {
            return false;
        }
        i += 2 + len;
    }
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

    /// The JPEG path decodes as `image::load_from_memory` + `into_rgb8` does: colour and greyscale, files from
    /// another encoder with 4:2:0, 4:2:2 and 4:4:4 chroma, progressive, and CMYK (`tests/fixtures/jpeg`, written by
    /// Pillow), whole and cut short (a cut scan still decodes), and fails where it fails.
    #[test]
    fn jpeg_decodes_as_the_generic_path() {
        use image::ImageEncoder;
        let generic = |bytes: &[u8]| {
            image::load_from_memory(bytes).map(|d| {
                let rgb = d.into_rgb8();
                (rgb.width(), rgb.height(), rgb.into_raw())
            })
        };
        let mut img = Image::new(61, 37, [0, 0, 0]);
        for (i, v) in img.data.iter_mut().enumerate() {
            *v = (i * 37 % 251) as u8;
        }
        let luma: Vec<u8> = img.data.chunks(3).map(|p| p[1]).collect();
        let mut files =
            vec![encode(&img, ImageFormat::Jpeg(90)).unwrap(), encode(&img, ImageFormat::Jpeg(100)).unwrap()];
        let mut grey = Vec::new();
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut grey, 85)
            .write_image(&luma, 61, 37, image::ExtendedColorType::L8)
            .unwrap();
        files.push(grey);
        files.extend(
            [
                &include_bytes!("../tests/fixtures/jpeg/s420.jpg")[..],
                include_bytes!("../tests/fixtures/jpeg/s422.jpg"),
                include_bytes!("../tests/fixtures/jpeg/s444.jpg"),
                include_bytes!("../tests/fixtures/jpeg/progressive420.jpg"),
                include_bytes!("../tests/fixtures/jpeg/cmyk.jpg"),
            ]
            .map(<[u8]>::to_vec),
        );
        for f in files {
            for bytes in [&f[..], &f[..f.len() * 2 / 3], &f[..f.len() / 4], &f[..3]] {
                match (decode_jpeg(bytes).unwrap(), generic(bytes)) {
                    (Ok(a), Ok((w, h, data))) => assert_eq!((a.width, a.height, a.data), (w, h, data)),
                    (Err(a), Err(b)) => assert_eq!(a.to_string(), b.to_string()),
                    (a, b) => {
                        panic!("{} of {} bytes: {:?} against {:?}", bytes.len(), f.len(), a.err(), b.err())
                    }
                }
            }
        }
        assert!(decode_jpeg(&encode(&img, ImageFormat::Png).unwrap()).is_none());
    }

    #[test]
    fn truncated_jpegs_are_detected() {
        let mut img = Image::new(64, 48, [0, 0, 0]);
        for y in 0..48 {
            for x in 0..64 {
                img.set_pixel(x, y, [(x * 4) as u8, (y * 5) as u8, ((x ^ y) * 3) as u8]);
            }
        }
        let jpg = encode(&img, ImageFormat::Jpeg(90)).unwrap();
        assert!(!jpeg_truncated(&jpg));
        // Trailing bytes after EOI (camera trailers) are fine.
        let mut trailer = jpg.clone();
        trailer.extend_from_slice(b"trailer");
        assert!(!jpeg_truncated(&trailer));
        // Cut mid-scan: the decoder still accepts it, the marker walk does not.
        let cut = &jpg[..jpg.len() * 3 / 4];
        assert!(jpeg_truncated(cut));
        // Cut inside the headers.
        assert!(jpeg_truncated(&jpg[..10]));
        let png = encode(&img, ImageFormat::Png).unwrap();
        assert!(!jpeg_truncated(&png));
        assert!(!jpeg_truncated(b"not an image"));
    }

    #[test]
    fn orientations_match_reference_across_sizes_and_tiles() {
        fn ref_oriented(img: &Image, o: u8) -> Image {
            let (w, h) = (img.width as usize, img.height as usize);
            let (ow, oh) = if o >= 5 { (h, w) } else { (w, h) };
            let mut out = vec![0u8; ow * oh * 3];
            for y in 0..oh {
                for x in 0..ow {
                    let (sx, sy) = match o {
                        2 => (w - 1 - x, y),
                        3 => (w - 1 - x, h - 1 - y),
                        4 => (x, h - 1 - y),
                        5 => (y, x),
                        6 => (y, h - 1 - x),
                        7 => (w - 1 - y, h - 1 - x),
                        _ => (w - 1 - y, x),
                    };
                    let s = (sy * w + sx) * 3;
                    let d = (y * ow + x) * 3;
                    out[d..d + 3].copy_from_slice(&img.data[s..s + 3]);
                }
            }
            Image { width: ow as u32, height: oh as u32, data: out }
        }
        for &(w, h) in &[(1u32, 1u32), (5, 7), (31, 33), (32, 32), (97, 67), (257, 193)] {
            let mut img = Image::new(w, h, [0, 0, 0]);
            for (i, v) in img.data.iter_mut().enumerate() {
                *v = (i.wrapping_mul(2654435761) >> 11) as u8;
            }
            for o in 1..=8u8 {
                let got = img.oriented(o);
                let want = if o == 1 { img.clone() } else { ref_oriented(&img, o) };
                assert_eq!(got, want, "oriented({o}) {w}x{h}");
                let serial = crate::par::serial(|| img.oriented(o));
                assert_eq!(serial, want, "oriented({o}) serial {w}x{h}");
            }
        }
    }

    #[test]
    fn png_decodes_as_the_generic_path() {
        use image::{ExtendedColorType, ImageEncoder};
        let generic = |bytes: &[u8]| {
            image::load_from_memory(bytes).map(|d| {
                let rgb = d.into_rgb8();
                (rgb.width(), rgb.height(), rgb.into_raw())
            })
        };
        let (w, h) = (73u32, 41u32);
        let n = (w * h) as usize;
        let rgb: Vec<u8> = (0..n * 3).map(|i| (i.wrapping_mul(37) % 251) as u8).collect();
        let rgba: Vec<u8> = (0..n * 4).map(|i| (i.wrapping_mul(53) % 251) as u8).collect();
        let luma: Vec<u8> = (0..n).map(|i| (i.wrapping_mul(29) % 251) as u8).collect();
        let la: Vec<u8> = (0..n * 2).map(|i| (i.wrapping_mul(19) % 251) as u8).collect();
        let rgb16: Vec<u8> = (0..n * 6).map(|i| (i.wrapping_mul(41) % 251) as u8).collect();

        let encode_png = |buf: &[u8], ct: ExtendedColorType| {
            let mut out = Vec::new();
            image::codecs::png::PngEncoder::new(&mut out).write_image(buf, w, h, ct).unwrap();
            out
        };
        let files = [
            encode_png(&rgb, ExtendedColorType::Rgb8),
            encode_png(&rgba, ExtendedColorType::Rgba8),
            encode_png(&luma, ExtendedColorType::L8),
            encode_png(&la, ExtendedColorType::La8),
            encode_png(&rgb16, ExtendedColorType::Rgb16),
        ];
        for f in &files {
            for bytes in [&f[..], &f[..f.len() * 2 / 3], &f[..12]] {
                match (decode_png(bytes).unwrap(), generic(bytes)) {
                    (Ok(a), Ok((gw, gh, gdata))) => assert_eq!((a.width, a.height, a.data), (gw, gh, gdata)),
                    (Err(a), Err(b)) => assert_eq!(a.to_string(), b.to_string()),
                    (a, b) => panic!("{} of {} bytes: {:?} vs {:?}", bytes.len(), f.len(), a.err(), b.err()),
                }
            }
        }
        assert!(decode_png(&encode(&grid(), ImageFormat::Jpeg(90)).unwrap()).is_none());
    }

    #[test]
    fn non_grey_fill_and_inverted_fill_rect_edge_cases() {
        for &(w, h) in &[(0u32, 0u32), (1, 1), (3, 1), (1, 3), (17, 19), (65, 33)] {
            let img = Image::new(w, h, [11, 122, 233]);
            assert_eq!(img.data.len(), w as usize * h as usize * 3);
            assert!(img.data.as_chunks::<3>().0.iter().all(|&px| px == [11, 122, 233]), "{w}x{h}");
        }
        let mut img = Image::new(10, 8, [5, 6, 7]);
        let orig = img.clone();
        // Inverted or out-of-bounds rects are no-ops without panicking.
        img.fill_rect(8, 2, 3, 6, [255, 0, 0]);
        img.fill_rect(2, 6, 8, 2, [255, 0, 0]);
        img.fill_rect(-10, -5, -1, 4, [255, 0, 0]);
        img.fill_rect(20, 2, 30, 6, [255, 0, 0]);
        assert_eq!(img, orig);
        // Valid clipped grey and non-grey fills.
        img.fill_rect(-2, 1, 4, 3, [90, 90, 90]);
        img.fill_rect(6, 5, 15, 12, [10, 20, 30]);
        for y in 0..8 {
            for x in 0..10 {
                let want = if x < 4 && (1..3).contains(&y) {
                    [90, 90, 90]
                } else if x >= 6 && y >= 5 {
                    [10, 20, 30]
                } else {
                    [5, 6, 7]
                };
                assert_eq!(img.pixel(x, y), want, "({x},{y})");
            }
        }
    }
}
