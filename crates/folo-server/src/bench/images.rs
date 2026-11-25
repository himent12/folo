//! Serving a benchmark's images: the stored files, the variants every contestant is graded on
//! ([`folo_bench::conditions::render`]) and thumbnails of both. A URL's bytes never change (a benchmark is
//! frozen), so everything is sent as immutable; recent variants and decoded base images are kept in bounded
//! single-flight LRU caches in memory, variant thumbnails on disk.

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use axum::http::StatusCode;
use folo_bench::{Condition, conditions};
use image::RgbImage;

use super::state::{ImageRecord, Slot};
use crate::error::{Error, Result};
use crate::index::{read_image_with, write_atomic};
use crate::store::ImageData;
use crate::util::{Flight, fnv1a};

/// Bytes of rendered variant JPEGs kept in memory (most recently used first out).
const RENDER_CACHE_BYTES: usize = 64 << 20;
/// Bytes of decoded base RGB images kept in memory so rendering multiple conditions/severities of the same
/// benchmark image reads and decodes the source file once.
const DECODED_CACHE_BYTES: usize = 64 << 20;

/// A byte-bounded LRU cache with single-flight deduplication and prefix invalidation (`forget`).
struct FlightLru<V> {
    state: Mutex<FlightLruState<V>>,
    max_bytes: usize,
    size_of: fn(&V) -> usize,
}

struct FlightLruState<V> {
    lru: VecDeque<(String, Arc<V>)>,
    in_flight: HashMap<String, Arc<Flight<Result<Arc<V>>>>>,
}

impl<V> FlightLru<V> {
    fn new(max_bytes: usize, size_of: fn(&V) -> usize) -> Self {
        FlightLru {
            state: Mutex::new(FlightLruState { lru: VecDeque::new(), in_flight: HashMap::new() }),
            max_bytes,
            size_of,
        }
    }

    #[cfg(test)]
    fn get_cached(&self, key: &str) -> Option<Arc<V>> {
        let mut st = self.state.lock().unwrap_or_else(|p| p.into_inner());
        let i = st.lru.iter().position(|(k, _)| k == key)?;
        let hit = st.lru.remove(i).expect("found");
        let val = hit.1.clone();
        st.lru.push_front(hit);
        Some(val)
    }

    fn insert_lru(
        lru: &mut VecDeque<(String, Arc<V>)>,
        key: String,
        val: Arc<V>,
        max_bytes: usize,
        size_of: fn(&V) -> usize,
    ) {
        lru.retain(|(k, _)| *k != key);
        lru.push_front((key, val));
        let mut total = 0usize;
        let keep = lru.iter().take_while(|(_, v)| {
            total = total.saturating_add(size_of(v));
            total <= max_bytes
        });
        let keep = keep.count().max(1);
        lru.truncate(keep);
    }

    fn forget_prefix(&self, prefix: &str) {
        let mut st = self.state.lock().unwrap_or_else(|p| p.into_inner());
        st.lru.retain(|(k, _)| !k.starts_with(prefix));
        st.in_flight.retain(|k, _| !k.starts_with(prefix));
    }

    /// Returns a cached value or runs `compute` at most once among concurrent callers for `key` (distinct keys run
    /// in parallel). Panic-safe: if a leader panics, waiters wake and retry cleanly. If `forget_prefix` removed
    /// the flight while `compute` ran, the stale result is returned to its caller without repopulating the LRU.
    fn get_or_compute(&self, key: String, mut compute: impl FnMut() -> Result<Arc<V>>) -> Result<Arc<V>> {
        loop {
            let mine = {
                let mut st = self.state.lock().unwrap_or_else(|p| p.into_inner());
                if let Some(i) = st.lru.iter().position(|(k, _)| *k == key) {
                    let hit = st.lru.remove(i).expect("found");
                    let val = hit.1.clone();
                    st.lru.push_front(hit);
                    return Ok(val);
                }
                if let Some(flight) = st.in_flight.get(&key).cloned() {
                    drop(st);
                    if let Some(result) = flight.wait() {
                        return result;
                    }
                    continue;
                }
                let flight = Flight::new();
                st.in_flight.insert(key.clone(), flight.clone());
                flight
            };

            struct Cleanup<'a, V> {
                cache: &'a FlightLru<V>,
                key: String,
                flight: Arc<Flight<Result<Arc<V>>>>,
                done: bool,
            }
            impl<V> Drop for Cleanup<'_, V> {
                fn drop(&mut self) {
                    if !self.done {
                        let mut st = self.cache.state.lock().unwrap_or_else(|p| p.into_inner());
                        if st.in_flight.get(&self.key).is_some_and(|cur| Arc::ptr_eq(cur, &self.flight)) {
                            st.in_flight.remove(&self.key);
                        }
                        drop(st);
                        self.flight.abort();
                    }
                }
            }

            let mut guard = Cleanup { cache: self, key: key.clone(), flight: mine.clone(), done: false };
            let result = compute();
            {
                let mut st = self.state.lock().unwrap_or_else(|p| p.into_inner());
                let was_current = st.in_flight.get(&key).is_some_and(|cur| Arc::ptr_eq(cur, &mine));
                if was_current {
                    st.in_flight.remove(&key);
                    if let Ok(val) = &result {
                        Self::insert_lru(&mut st.lru, key, val.clone(), self.max_bytes, self.size_of);
                    }
                }
            }
            mine.finish(result.clone());
            guard.done = true;
            return result;
        }
    }
}

/// Bounded single-flight LRU caches for rendered stress variants and decoded base benchmark images.
pub struct Renders {
    variants: FlightLru<Vec<u8>>,
    decoded: FlightLru<RgbImage>,
}

impl Default for Renders {
    fn default() -> Self {
        Renders {
            variants: FlightLru::new(RENDER_CACHE_BYTES, Vec::len),
            decoded: FlightLru::new(DECODED_CACHE_BYTES, |img| img.as_raw().len()),
        }
    }
}

impl Renders {
    /// Drops everything of benchmark `bench` (it was deleted), across both rendered variants and decoded base
    /// images.
    pub fn forget(&self, bench: &str) {
        let prefix = format!("{bench}/");
        self.variants.forget_prefix(&prefix);
        self.decoded.forget_prefix(&prefix);
    }

    /// Decodes (or reuses from the bounded LRU) the upright RGB pixels of benchmark image `im`.
    fn base_image(&self, slot: &Slot, im: &ImageRecord) -> Result<Arc<RgbImage>> {
        let key = format!("{}/{}", slot.id, im.id);
        self.decoded.get_or_compute(key, || {
            let path = slot.image_path(im);
            let bytes = std::fs::read(&path).map_err(|e| missing(&path, e))?;
            let img = conditions::decode(&bytes).map_err(|e| {
                Error::new(StatusCode::UNPROCESSABLE_ENTITY, "image_decode_failed", e.to_string())
            })?;
            Ok(Arc::new(img))
        })
    }

    /// The canonical JPEG of image `im` of `slot` under `condition` at `severity` (`seed` is the benchmark's).
    /// Concurrent requests for the same `(bench, image, condition, severity)` share one render; requests for
    /// different conditions/severities of the same image share one base-image decode.
    pub fn render(
        &self,
        slot: &Slot,
        im: &ImageRecord,
        condition: Condition,
        severity: u8,
        seed: u64,
    ) -> Result<Arc<Vec<u8>>> {
        if !(1..=folo_api::BENCH_SEVERITIES).contains(&severity) {
            return Err(Error::new(
                StatusCode::UNPROCESSABLE_ENTITY,
                "image_decode_failed",
                format!("severity must be 1..={}, got {severity}", folo_api::BENCH_SEVERITIES),
            ));
        }
        let key = format!("{}/{}/{}/{severity}", slot.id, im.id, condition.id());
        self.variants.get_or_compute(key, || {
            let base = self.base_image(slot, im)?;
            let seed = conditions::variant_seed(seed, &im.id, condition, severity);
            let (out, _) = conditions::apply(condition, severity, &base, &im.boxes, seed).map_err(|e| {
                Error::new(StatusCode::UNPROCESSABLE_ENTITY, "image_decode_failed", e.to_string())
            })?;
            let jpeg = conditions::encode_jpeg(&out).map_err(|e| {
                Error::new(StatusCode::UNPROCESSABLE_ENTITY, "image_decode_failed", e.to_string())
            })?;
            Ok(Arc::new(jpeg))
        })
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
    use std::sync::Barrier;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

    use super::*;

    #[test]
    fn the_render_cache_is_bounded_and_most_recent_first() {
        let r = Renders::default();
        let big = Arc::new(vec![0u8; RENDER_CACHE_BYTES / 2 + 1]);
        r.variants.get_or_compute("a/1".into(), || Ok(big.clone())).unwrap();
        r.variants.get_or_compute("a/2".into(), || Ok(big.clone())).unwrap();
        // Two halves do not fit: the older went.
        assert!(r.variants.get_cached("a/1").is_none() && r.variants.get_cached("a/2").is_some());
        r.variants.get_or_compute("b/1".into(), || Ok(Arc::new(vec![1, 2, 3]))).unwrap();
        assert!(r.variants.get_cached("a/2").is_some() && r.variants.get_cached("b/1").is_some());
        r.decoded.get_or_compute("a/img1".into(), || Ok(Arc::new(RgbImage::new(4, 4)))).unwrap();
        r.decoded.get_or_compute("b/img1".into(), || Ok(Arc::new(RgbImage::new(4, 4)))).unwrap();
        r.forget("a");
        assert!(r.variants.get_cached("a/2").is_none() && r.variants.get_cached("b/1").is_some());
        assert!(r.decoded.get_cached("a/img1").is_none() && r.decoded.get_cached("b/img1").is_some());
        // One entry larger than the cache is still kept, alone.
        r.variants.get_or_compute("c/1".into(), || Ok(Arc::new(vec![0u8; RENDER_CACHE_BYTES + 1]))).unwrap();
        assert!(r.variants.get_cached("c/1").is_some() && r.variants.get_cached("b/1").is_none());
    }

    /// Single-flight deduplicates concurrent renders for the same key, lets distinct keys run in parallel,
    /// recovers cleanly from a leader panic, and prevents forgotten flights from repopulating the LRU.
    #[test]
    fn flight_lru_deduplicates_runs_distinct_keys_in_parallel_and_recovers() {
        let cache = FlightLru::new(1024, Vec::len);
        let calls = AtomicUsize::new(0);
        let entered = Barrier::new(2);
        let release = Barrier::new(2);

        let (v1, v2) = std::thread::scope(|s| {
            let h1 = s.spawn(|| {
                cache
                    .get_or_compute("b1/i1/defocus/1".into(), || {
                        calls.fetch_add(1, Ordering::SeqCst);
                        entered.wait();
                        release.wait();
                        Ok(Arc::new(vec![7u8; 4]))
                    })
                    .unwrap()
            });
            entered.wait();
            let h2 = s.spawn(|| {
                cache
                    .get_or_compute("b1/i1/defocus/1".into(), || {
                        calls.fetch_add(1, Ordering::SeqCst);
                        Ok(Arc::new(vec![9u8; 4]))
                    })
                    .unwrap()
            });
            let deadline = Instant::now() + Duration::from_secs(5);
            let mut parked = false;
            while Instant::now() < deadline {
                if cache.state.lock().unwrap().in_flight["b1/i1/defocus/1"].waiters() >= 1 {
                    parked = true;
                    break;
                }
                std::thread::yield_now();
            }
            release.wait();
            let out = (h1.join().unwrap(), h2.join().unwrap());
            assert!(parked, "waiter did not park in Flight::wait");
            out
        });
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert!(Arc::ptr_eq(&v1, &v2));

        // Distinct keys run concurrently.
        let both = crate::util::Rendezvous::new();
        std::thread::scope(|s| {
            let handles: Vec<_> = ["b1/i1/haze/1", "b1/i1/haze/2"]
                .into_iter()
                .map(|k| {
                    let (cache, both) = (&cache, &both);
                    s.spawn(move || {
                        let mut met = false;
                        cache
                            .get_or_compute(k.into(), || {
                                met = both.arrive(Duration::from_secs(5));
                                Ok(Arc::new(vec![1]))
                            })
                            .unwrap();
                        assert!(met, "distinct render keys did not run concurrently");
                    })
                })
                .collect();
            for h in handles {
                h.join().unwrap();
            }
        });

        // Leader panic wakes waiter to retry cleanly.
        let entered = Barrier::new(2);
        let release = Barrier::new(2);
        let boom = AtomicBool::new(true);
        std::thread::scope(|s| {
            let h1 = s.spawn(|| {
                let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let _ = cache.get_or_compute("b1/i1/panic/1".into(), || {
                        entered.wait();
                        release.wait();
                        if boom.swap(false, Ordering::SeqCst) {
                            panic!("render boom");
                        }
                        Ok(Arc::new(vec![0]))
                    });
                }));
            });
            entered.wait();
            let h2 =
                s.spawn(|| cache.get_or_compute("b1/i1/panic/1".into(), || Ok(Arc::new(vec![42]))).unwrap());
            let deadline = Instant::now() + Duration::from_secs(5);
            let mut parked = false;
            while Instant::now() < deadline {
                if cache.state.lock().unwrap().in_flight["b1/i1/panic/1"].waiters() >= 1 {
                    parked = true;
                    break;
                }
                std::thread::yield_now();
            }
            release.wait();
            h1.join().unwrap();
            let got = h2.join().unwrap();
            assert!(parked, "waiter did not park before leader panic");
            assert_eq!(*got, vec![42]);
        });

        // `forget_prefix` while in flight prevents stale result from entering LRU.
        let entered = Barrier::new(2);
        let release = Barrier::new(2);
        std::thread::scope(|s| {
            let h = s.spawn(|| {
                cache
                    .get_or_compute("b2/i1/blur/1".into(), || {
                        entered.wait();
                        release.wait();
                        Ok(Arc::new(vec![5]))
                    })
                    .unwrap()
            });
            entered.wait();
            cache.forget_prefix("b2/");
            release.wait();
            assert_eq!(*h.join().unwrap(), vec![5]);
        });
        assert!(cache.get_cached("b2/i1/blur/1").is_none());
    }

    /// Rendering via the decoded base-image cache produces bytes identical to `folo_bench::conditions::render`.
    #[test]
    fn decoded_base_cache_matches_conditions_render_byte_for_byte() {
        let mut rgb = RgbImage::new(32, 24);
        for (x, y, p) in rgb.enumerate_pixels_mut() {
            *p = image::Rgb([(x * 7) as u8, (y * 10) as u8, ((x + y) * 3) as u8]);
        }
        let raw_jpeg = conditions::encode_jpeg(&rgb).unwrap();
        let boxes = vec![folo_api::BBox { class: 0, x1: 2.0, y1: 3.0, x2: 20.0, y2: 18.0 }];
        let seed = conditions::variant_seed(42, "0", Condition::Defocus, 2);

        let expected = conditions::render(&raw_jpeg, Condition::Defocus, 2, &boxes, seed).unwrap().jpeg;
        let decoded = conditions::decode(&raw_jpeg).unwrap();
        let (out, _) = conditions::apply(Condition::Defocus, 2, &decoded, &boxes, seed).unwrap();
        let actual = conditions::encode_jpeg(&out).unwrap();
        assert_eq!(actual, expected);
    }
}
