//! `--cache ram`: every training image decoded once and kept resized (long side `imgsz`, as the transform resizes
//! it), so later epochs skip the JPEG decode and the resize (about 89% of a mosaic sample for 1080p JPEGs).
//!
//! The cache holds exactly what [`crate::augment::resize_long_side`] makes of the decoded image, plus the decoded
//! size. The transform's own resize of such an image is the identity (rows are copied), and its letterbox
//! geometry is computed from the original size ([`crate::augment::Fetched::orig_size`]), so a batch is the same, pixel
//! for pixel and box for box, with or without the cache. Images are cached the first time a worker loads them
//! (any order: the result does not depend on it); an image that cannot be read is not cached.

use crate::augment::resize_long_side;
use crate::dataset::Dataset;
use crate::error::{DataError, Result};
use crate::image::Sample;
use std::sync::{Arc, OnceLock};

/// How [`crate::DataLoader`] keeps images between epochs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CacheMode {
    /// Decode every image every time.
    #[default]
    None,
    /// Keep every image decoded and resized in memory (`--cache ram`).
    Ram,
}

/// Part of the available memory a RAM cache may take.
pub const RAM_CACHE_SHARE: f64 = 0.8;

/// A cached image: the sample resized, and its decoded size.
type Entry = Arc<(Sample, [u32; 2])>;

/// Images decoded and resized once, by dataset index.
pub(crate) struct RamCache {
    imgsz: u32,
    slots: Vec<OnceLock<Entry>>,
}

/// The size of an image after [`resize_long_side`] to `long`.
fn resized(w: u32, h: u32, long: u32) -> (u64, u64) {
    let r = long as f64 / w.max(h) as f64;
    (((w as f64 * r).round() as u64).max(1), ((h as f64 * r).round() as u64).max(1))
}

/// Bytes a RAM cache of `ds` at `imgsz` holds: every image resized (its size from the dataset when known, else a
/// full `imgsz x imgsz`), 3 bytes a pixel, plus its boxes.
pub fn ram_cache_bytes(ds: &dyn Dataset, imgsz: u32) -> u64 {
    (0..ds.len())
        .map(|i| {
            let (w, h) = ds.image_size(i).filter(|&(w, h)| w > 0 && h > 0).unwrap_or((imgsz, imgsz));
            let (rw, rh) = resized(w, h, imgsz);
            rw * rh * 3 + 20 * ds.boxes(i).len() as u64 + 64
        })
        .sum()
}

/// Memory the system can still give, in bytes.
fn available_memory() -> u64 {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    sys.available_memory()
}

fn gib(b: u64) -> String {
    format!("{:.1} GiB", b as f64 / (1u64 << 30) as f64)
}

impl RamCache {
    /// A cache for `ds` at `imgsz`, after checking that it fits in [`RAM_CACHE_SHARE`] of the available memory
    /// (an error saying how much it needs otherwise).
    pub fn new(ds: &dyn Dataset, imgsz: u32) -> Result<Self> {
        let need = ram_cache_bytes(ds, imgsz);
        let free = available_memory();
        if need as f64 > RAM_CACHE_SHARE * free as f64 {
            return Err(DataError::Config(format!(
                "--cache ram needs {} for {} images at imgsz {imgsz}, more than {:.0}% of the {} of memory available; \
                 train without --cache, with a smaller --imgsz, or on fewer images",
                gib(need),
                ds.len(),
                RAM_CACHE_SHARE * 100.0,
                gib(free)
            )));
        }
        Ok(RamCache { imgsz, slots: (0..ds.len()).map(|_| OnceLock::new()).collect() })
    }

    /// Sample `i` resized, and its decoded size: from the cache, else loaded with `load` and kept.
    pub fn get(&self, i: usize, load: impl FnOnce() -> Result<Sample>) -> Result<(Sample, [u32; 2])> {
        let slot = &self.slots[i];
        let entry = match slot.get() {
            Some(e) => e.clone(),
            None => {
                let s = load()?;
                let orig = [s.image.width, s.image.height];
                // Two workers may race on one image: both resize, one is kept; the results are identical.
                slot.get_or_init(|| Arc::new((resize_long_side(s, self.imgsz), orig))).clone()
            }
        };
        Ok((entry.0.clone(), entry.1))
    }
}
