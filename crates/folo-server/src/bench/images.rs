//! Serving a benchmark's images: the stored files, the variants every contestant is graded on
//! ([`folo_bench::conditions::render`]) and thumbnails of both. A URL's bytes never change (a benchmark is
//! frozen), so everything is sent as immutable; recent variants are kept in memory, variant thumbnails on disk.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use axum::http::StatusCode;
use folo_bench::{Condition, conditions};

use super::state::{ImageRecord, Slot};
use crate::error::{Error, Result};
use crate::index::{read_image_with, write_atomic};
use crate::store::ImageData;
use crate::util::fnv1a;

/// Bytes of rendered variants kept in memory (most recently used first out).
const RENDER_CACHE_BYTES: usize = 64 << 20;

/// A small LRU of rendered variants, keyed by benchmark, image, condition and severity.
#[derive(Default)]
pub struct Renders {
    cache: Mutex<VecDeque<(String, Arc<Vec<u8>>)>>,
}

impl Renders {
    fn get(&self, key: &str) -> Option<Arc<Vec<u8>>> {
        let mut cache = self.cache.lock().unwrap();
        let i = cache.iter().position(|(k, _)| k == key)?;
        let hit = cache.remove(i).expect("found");
        let bytes = hit.1.clone();
        cache.push_front(hit);
        Some(bytes)
    }

    fn put(&self, key: String, bytes: Arc<Vec<u8>>) {
        let mut cache = self.cache.lock().unwrap();
        cache.retain(|(k, _)| *k != key);
        cache.push_front((key, bytes));
        let mut total = 0usize;
        let keep = cache.iter().take_while(|(_, b)| {
            total += b.len();
            total <= RENDER_CACHE_BYTES
        });
        let keep = keep.count().max(1);
        cache.truncate(keep);
    }

    /// Drops everything of benchmark `bench` (it was deleted).
    pub fn forget(&self, bench: &str) {
        let prefix = format!("{bench}/");
        self.cache.lock().unwrap().retain(|(k, _)| !k.starts_with(&prefix));
    }

    /// The canonical JPEG of image `im` of `slot` under `condition` at `severity` (`seed` is the benchmark's).
    pub fn render(
        &self,
        slot: &Slot,
        im: &ImageRecord,
        condition: Condition,
        severity: u8,
        seed: u64,
    ) -> Result<Arc<Vec<u8>>> {
        let key = format!("{}/{}/{}/{severity}", slot.id, im.id, condition.id());
        if let Some(hit) = self.get(&key) {
            return Ok(hit);
        }
        let path = slot.image_path(im);
        let bytes = std::fs::read(&path).map_err(|e| missing(&path, e))?;
        let seed = conditions::variant_seed(seed, &im.id, condition, severity);
        let rendered = conditions::render(&bytes, condition, severity, &im.boxes, seed).map_err(|e| {
            Error::new(StatusCode::UNPROCESSABLE_ENTITY, "image_decode_failed", e.to_string())
        })?;
        let jpeg = Arc::new(rendered.jpeg);
        self.put(key, jpeg.clone());
        Ok(jpeg)
    }
}

fn missing(path: &Path, e: std::io::Error) -> Error {
    if e.kind() == std::io::ErrorKind::NotFound {
        Error::new(StatusCode::NOT_FOUND, "image_not_found", format!("{} no longer exists", path.display()))
    } else {
        Error::internal(format!("reading {}: {e}", path.display()))
    }
}

/// Image `im` of `slot`: the stored file, or a JPEG thumbnail whose long side is `thumb`; with `variant`, the
/// image under that condition (or its thumbnail). Thumbnails are cached in `thumbs` (the benchmark's folder of
/// `<data dir>/cache/thumbs`), unless the benchmark was deleted meanwhile.
pub fn image(
    renders: &Renders,
    slot: &Slot,
    im: &ImageRecord,
    seed: u64,
    variant: Option<(Condition, u8)>,
    thumb: Option<u32>,
    thumbs: &Path,
) -> Result<ImageData> {
    let save = |file: &Path, bytes: &[u8]| {
        if !slot.lock().deleted
            && let Err(e) = write_atomic(file, bytes)
        {
            tracing::warn!("caching thumbnail {}: {e}", file.display());
        }
    };
    let Some((condition, severity)) = variant else {
        let data = read_image_with(&slot.image_path(im), thumb, thumb.map(|_| thumbs), save)?;
        return Ok(ImageData { immutable: true, ..data });
    };
    let Some(side) = thumb else {
        let jpeg = renders.render(slot, im, condition, severity, seed)?;
        return Ok(ImageData { content_type: "image/jpeg", bytes: jpeg.to_vec(), immutable: true });
    };
    // A variant's thumbnail: its file never changes, so the key needs no file time.
    let key = format!("{}|{}|{}|{severity}|{side}", slot.id, im.id, condition.id());
    let cached: PathBuf = thumbs.join(format!("v{:016x}.jpg", fnv1a(key.as_bytes())));
    if let Ok(bytes) = std::fs::read(&cached) {
        return Ok(ImageData { content_type: "image/jpeg", bytes, immutable: true });
    }
    let jpeg = renders.render(slot, im, condition, severity, seed)?;
    let img = folo_data::decode(&jpeg).map_err(|e| Error::internal(e.to_string()))?;
    let long = img.width.max(img.height);
    let img = if side < long {
        let scale = side as f64 / long as f64;
        let w = ((img.width as f64 * scale).round() as u32).max(1);
        let h = ((img.height as f64 * scale).round() as u32).max(1);
        folo_data::resize::resize(&img, w, h)
    } else {
        img
    };
    let bytes = folo_data::encode(&img, folo_data::ImageFormat::Jpeg(85))
        .map_err(|e| Error::internal(e.to_string()))?;
    save(&cached, &bytes);
    Ok(ImageData { content_type: "image/jpeg", bytes, immutable: true })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_render_cache_is_bounded_and_most_recent_first() {
        let r = Renders::default();
        let big = Arc::new(vec![0u8; RENDER_CACHE_BYTES / 2 + 1]);
        r.put("a/1".into(), big.clone());
        r.put("a/2".into(), big.clone());
        // Two halves do not fit: the older went.
        assert!(r.get("a/1").is_none() && r.get("a/2").is_some());
        r.put("b/1".into(), Arc::new(vec![1, 2, 3]));
        assert!(r.get("a/2").is_some() && r.get("b/1").is_some());
        r.forget("a");
        assert!(r.get("a/2").is_none() && r.get("b/1").is_some());
        // One entry larger than the cache is still kept, alone.
        r.put("c/1".into(), Arc::new(vec![0u8; RENDER_CACHE_BYTES + 1]));
        assert!(r.get("c/1").is_some() && r.get("b/1").is_none());
    }
}
