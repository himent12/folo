//! Multi-threaded, deterministic batching data loader.
//!
//! Each epoch has a fixed sample order (a seeded shuffle of `(seed, epoch)`) and every sample position has its own
//! RNG stream `(seed, epoch, position)`. Worker threads claim positions from a shared counter, decode and augment,
//! and send results to a collator thread that restores the order and builds [`Batch`]es. The output is therefore
//! identical for any number of workers. In-flight work is bounded by a window of positions plus a bounded batch
//! channel; dropping the epoch iterator stops and joins all threads.

use crate::augment::{AugmentConfig, Fetched, TrainTransform, Transformed};
use crate::dataset::{Dataset, Issue, IssueKind};
use crate::error::{DataError, Result};
use crate::par;
use crate::ram_cache::{CacheMode, RamCache};
use crate::rng::Rng;
use crossbeam_channel::{Receiver, bounded, unbounded};
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;

/// Loader settings.
#[derive(Clone, Debug, PartialEq)]
pub struct LoaderConfig {
    /// Images per batch.
    pub batch_size: usize,
    /// Square output size; a multiple of 32 is typical.
    pub imgsz: u32,
    /// Worker threads. `0` decodes on the consumer thread (useful for debugging); results do not depend on it.
    pub workers: usize,
    /// Shuffle the order each epoch.
    pub shuffle: bool,
    /// Drop the last incomplete batch.
    pub drop_last: bool,
    /// Seed for shuffling and augmentation.
    pub seed: u64,
    /// Augmentation; `None` for validation (letterbox only).
    pub augment: Option<AugmentConfig>,
    /// Total epochs of the run, used for `close_mosaic` (0 = unknown, mosaic always on).
    pub epochs: u32,
    /// Finished batches buffered ahead of the consumer.
    pub prefetch_batches: usize,
    /// Deliver the pixels as `u8` ([`Batch::images_u8`], a quarter of the bytes, for uploading to a GPU that
    /// converts them) instead of `f32` in `0..1` ([`Batch::images`]).
    pub u8_images: bool,
    /// Keep images between epochs ([`CacheMode::Ram`]: decoded and resized, `--cache ram`). Batches are the same
    /// either way.
    pub cache: CacheMode,
}

impl Default for LoaderConfig {
    fn default() -> Self {
        LoaderConfig {
            batch_size: 16,
            imgsz: 640,
            workers: std::thread::available_parallelism().map_or(4, |n| n.get().min(16)),
            shuffle: true,
            drop_last: false,
            seed: 0,
            augment: Some(AugmentConfig::default()),
            epochs: 0,
            prefetch_batches: 2,
            u8_images: false,
            cache: CacheMode::None,
        }
    }
}

impl LoaderConfig {
    /// A validation loader config: no shuffle, no augmentation, keep the last batch.
    pub fn val(batch_size: usize, imgsz: u32) -> Self {
        LoaderConfig { batch_size, imgsz, shuffle: false, augment: None, ..LoaderConfig::default() }
    }
}

/// Where an image in a batch came from and how it was letterboxed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImageMeta {
    /// Index in the dataset of the image actually used (an unreadable image is replaced by the next readable one).
    pub index: usize,
    /// Original `[width, height]` as decoded (after EXIF orientation).
    pub orig_size: [u32; 2],
    /// Letterbox scale `[x, y]` from original to network pixels (validation only; see [`Transformed`]).
    pub ratio: [f32; 2],
    /// Letterbox padding `[left, top]` in network pixels.
    pub pad: [f32; 2],
}

impl ImageMeta {
    /// Maps network-pixel corners `[x1, y1, x2, y2]` back to original pixels (clipped).
    pub fn unmap_xyxy(&self, [x1, y1, x2, y2]: [f32; 4]) -> [f32; 4] {
        let ([rx, ry], [px, py]) = (self.ratio, self.pad);
        let (w, h) = (self.orig_size[0] as f32, self.orig_size[1] as f32);
        [
            ((x1 - px) / rx).clamp(0.0, w),
            ((y1 - py) / ry).clamp(0.0, h),
            ((x2 - px) / rx).clamp(0.0, w),
            ((y2 - py) / ry).clamp(0.0, h),
        ]
    }
}

/// A batch ready for the network.
#[derive(Clone, Debug, PartialEq)]
pub struct Batch {
    /// `n * 3 * h * w` floats in `0..1`, `NCHW`, contiguous, channels `R G B`; empty with
    /// [`LoaderConfig::u8_images`].
    pub images: Vec<f32>,
    /// With [`LoaderConfig::u8_images`]: the same pixels as `u8` (`NCHW`, `R G B`, `value / 255` is the float);
    /// empty otherwise.
    pub images_u8: Vec<u8>,
    /// `[n, 3, h, w]`.
    pub shape: [usize; 4],
    /// One row per box: `[batch_idx, class, cx, cy, w, h]`, coordinates normalised to the network input.
    pub targets: Vec<[f32; 6]>,
    /// Per image: dataset index, original size, letterbox ratio and pad.
    pub meta: Vec<ImageMeta>,
}

impl Batch {
    /// Number of images.
    pub fn len(&self) -> usize {
        self.shape[0]
    }

    /// Whether the batch has no images.
    pub fn is_empty(&self) -> bool {
        self.shape[0] == 0
    }
}

/// Throughput of an epoch so far.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Throughput {
    /// Images delivered to the consumer.
    pub images: usize,
    /// Batches delivered.
    pub batches: usize,
    /// Seconds since the epoch started.
    pub seconds: f64,
    /// `images / seconds`.
    pub images_per_s: f64,
}

struct Shared {
    ds: Arc<dyn Dataset>,
    cfg: LoaderConfig,
    failures: Mutex<BTreeMap<usize, String>>,
    /// With [`CacheMode::Ram`].
    ram: Option<RamCache>,
}

/// Maximum consecutive substitutes tried for an unreadable image.
const MAX_SUBSTITUTES: usize = 64;

impl Shared {
    /// Loads sample `i`; if it cannot be read, records the failure and deterministically tries `i + 1`, `i + 2`, ...
    fn fetch(&self, i: usize) -> Result<Fetched> {
        let n = self.ds.len();
        let mut last = None;
        for k in 0..MAX_SUBSTITUTES.min(n) {
            let j = (i + k) % n;
            let got = match &self.ram {
                Some(c) => c.get(j, || self.ds.get(j)).map(|(sample, orig_size)| Fetched {
                    index: j,
                    sample,
                    orig_size,
                }),
                None => self.ds.get(j).map(|s| Fetched::decoded(j, s)),
            };
            match got {
                Ok(f) => return Ok(f),
                Err(e) => {
                    self.failures
                        .lock()
                        .unwrap_or_else(|p| p.into_inner())
                        .entry(j)
                        .or_insert_with(|| e.to_string());
                    last = Some(e);
                }
            }
        }
        Err(last.unwrap_or(DataError::OutOfRange { index: i, len: n }))
    }

    /// Transforms the sample at epoch position `pos`. A panic (a bug in a dataset or transform) becomes an error
    /// for that batch instead of silently ending the epoch.
    fn produce(
        &self,
        transform: &TrainTransform,
        epoch: u32,
        pos: usize,
        index: usize,
    ) -> Result<Transformed> {
        let mut rng = Rng::derive(&[self.cfg.seed, epoch as u64, pos as u64]);
        let run = || transform.apply(index, self.ds.len(), &|i| self.fetch(i), &mut rng);
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)).unwrap_or_else(|p| {
            let msg = p
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| p.downcast_ref::<String>().cloned());
            Err(DataError::Format(format!("panic while loading sample {index}: {}", msg.unwrap_or_default())))
        })
    }
}

/// Batches a [`Dataset`] with worker threads. See the module docs for the determinism guarantees.
pub struct DataLoader {
    shared: Arc<Shared>,
}

impl DataLoader {
    /// Creates a loader. Fails on an empty dataset or invalid settings.
    pub fn new(ds: Arc<dyn Dataset>, cfg: LoaderConfig) -> Result<Self> {
        if cfg.batch_size == 0 {
            return Err(DataError::Config("batch_size must be at least 1".into()));
        }
        if cfg.imgsz < 8 {
            return Err(DataError::Config(format!("imgsz must be at least 8, got {}", cfg.imgsz)));
        }
        if let Some(a) = &cfg.augment {
            a.validate().map_err(DataError::Config)?;
        }
        if ds.is_empty() {
            return Err(DataError::Config("dataset is empty".into()));
        }
        let ram = match cfg.cache {
            CacheMode::Ram => Some(RamCache::new(&*ds, cfg.imgsz)?),
            CacheMode::None => None,
        };
        Ok(DataLoader { shared: Arc::new(Shared { ds, cfg, failures: Mutex::new(BTreeMap::new()), ram }) })
    }

    /// The configuration.
    pub fn config(&self) -> &LoaderConfig {
        &self.shared.cfg
    }

    /// The dataset.
    pub fn dataset(&self) -> &Arc<dyn Dataset> {
        &self.shared.ds
    }

    /// Batches per epoch.
    pub fn batches_per_epoch(&self) -> usize {
        let (n, b) = (self.shared.ds.len(), self.shared.cfg.batch_size);
        if self.shared.cfg.drop_last { n / b } else { n.div_ceil(b) }
    }

    /// Images that could not be read so far (each was replaced by the next readable one).
    pub fn failures(&self) -> Vec<Issue> {
        let f = self.shared.failures.lock().unwrap_or_else(|p| p.into_inner());
        f.iter()
            .map(|(&i, msg)| Issue {
                kind: IssueKind::CorruptImage,
                path: self
                    .shared
                    .ds
                    .image_path(i)
                    .map(PathBuf::from)
                    .unwrap_or_else(|| PathBuf::from(format!("#{i}"))),
                message: msg.clone(),
            })
            .collect()
    }

    /// The sample order of `epoch`.
    pub fn order(&self, epoch: u32) -> Vec<usize> {
        let n = self.shared.ds.len();
        let mut order: Vec<usize> = (0..n).collect();
        if self.shared.cfg.shuffle {
            Rng::derive(&[self.shared.cfg.seed, epoch as u64, u64::MAX]).shuffle(&mut order);
        }
        let keep = self.batches_per_epoch() * self.shared.cfg.batch_size;
        order.truncate(keep.min(n));
        order
    }

    /// Starts `epoch` (0-based) and returns an iterator over its batches.
    pub fn epoch(&self, epoch: u32) -> EpochIter {
        let cfg = &self.shared.cfg;
        let transform = match cfg.augment {
            Some(a) => TrainTransform::train(cfg.imgsz, a, epoch, cfg.epochs),
            None => TrainTransform::val(cfg.imgsz),
        };
        let order = Arc::new(self.order(epoch));
        let num_batches = order.len().div_ceil(cfg.batch_size);
        let mut it = EpochIter {
            started: Instant::now(),
            images: 0,
            batches: 0,
            num_batches,
            inner: Inner::Sync { shared: self.shared.clone(), transform, order: order.clone(), epoch },
        };
        if cfg.workers > 0 {
            it.inner = spawn(self.shared.clone(), transform, order, epoch, num_batches);
        }
        it
    }
}

/// Window gate: workers may claim position `p` only while `p < done + window`.
struct Gate {
    state: Mutex<GateState>,
    cv: Condvar,
    stop: AtomicBool,
}

struct GateState {
    next: usize,
    done: usize,
}

impl Gate {
    fn claim(&self, total: usize, window: usize) -> Option<usize> {
        let mut st = self.state.lock().unwrap_or_else(|p| p.into_inner());
        loop {
            if self.stop.load(Ordering::Acquire) || st.next >= total {
                return None;
            }
            if st.next < st.done + window {
                st.next += 1;
                return Some(st.next - 1);
            }
            st = self.cv.wait(st).unwrap_or_else(|p| p.into_inner());
        }
    }

    fn advance(&self, done: usize) {
        self.state.lock().unwrap_or_else(|p| p.into_inner()).done = done;
        self.cv.notify_all();
    }

    fn shutdown(&self) {
        self.stop.store(true, Ordering::Release);
        let _guard = self.state.lock().unwrap_or_else(|p| p.into_inner());
        self.cv.notify_all();
    }
}

enum Inner {
    Sync { shared: Arc<Shared>, transform: TrainTransform, order: Arc<Vec<usize>>, epoch: u32 },
    Threads { rx: Option<Receiver<Result<Batch>>>, gate: Arc<Gate>, threads: Vec<JoinHandle<()>> },
}

fn spawn(
    shared: Arc<Shared>,
    transform: TrainTransform,
    order: Arc<Vec<usize>>,
    epoch: u32,
    num_batches: usize,
) -> Inner {
    let cfg = &shared.cfg;
    let total = order.len();
    let window = cfg.batch_size * (cfg.prefetch_batches + 1) + cfg.workers;
    let gate = Arc::new(Gate {
        state: Mutex::new(GateState { next: 0, done: 0 }),
        cv: Condvar::new(),
        stop: false.into(),
    });
    let (sample_tx, sample_rx) = unbounded::<(usize, Result<Transformed>)>();
    let (batch_tx, batch_rx) = bounded::<Result<Batch>>(cfg.prefetch_batches.max(1));
    let mut threads = Vec::with_capacity(cfg.workers + 1);
    for w in 0..cfg.workers {
        let (shared, order, gate, tx) = (shared.clone(), order.clone(), gate.clone(), sample_tx.clone());
        let handle = std::thread::Builder::new()
            .name(format!("folo-data-{w}"))
            .spawn(move || {
                par::serial(|| {
                    while let Some(pos) = gate.claim(total, window) {
                        let out = shared.produce(&transform, epoch, pos, order[pos]);
                        if tx.send((pos, out)).is_err() {
                            break;
                        }
                    }
                })
            })
            .expect("failed to spawn a data loader thread");
        threads.push(handle);
    }
    drop(sample_tx);
    let collator = {
        let (shared, gate) = (shared.clone(), gate.clone());
        std::thread::Builder::new()
            .name("folo-data-collate".into())
            .spawn(move || {
                let bs = shared.cfg.batch_size;
                let collate_serial = shared.cfg.workers <= 1;
                let mut pending: HashMap<usize, Result<Transformed>> = HashMap::new();
                for b in 0..num_batches {
                    let (lo, hi) = (b * bs, ((b + 1) * bs).min(total));
                    while (lo..hi).any(|p| !pending.contains_key(&p)) {
                        match sample_rx.recv() {
                            Ok((p, s)) => {
                                pending.insert(p, s);
                            }
                            Err(_) => return, // workers stopped
                        }
                    }
                    let items: Vec<Result<Transformed>> =
                        (lo..hi).map(|p| pending.remove(&p).unwrap()).collect();
                    let batch = if collate_serial {
                        par::serial(|| collate(items, &shared.cfg))
                    } else {
                        collate(items, &shared.cfg)
                    };
                    gate.advance(hi);
                    if batch_tx.send(batch).is_err() {
                        return;
                    }
                }
            })
            .expect("failed to spawn the data loader collator")
    };
    threads.push(collator);
    Inner::Threads { rx: Some(batch_rx), gate, threads }
}

/// Converts transformed samples into a [`Batch`].
fn collate(items: Vec<Result<Transformed>>, cfg: &LoaderConfig) -> Result<Batch> {
    let items: Vec<Transformed> = items.into_iter().collect::<Result<_>>()?;
    let s = cfg.imgsz as usize;
    let plane = s * s;
    let n = items.len();
    // Interleaved RGB → planar, through `conv` (the f32 LUT or the identity).
    fn planar<T: Copy + Send + Sync + Default>(
        items: &[Transformed],
        s: usize,
        conv: impl Fn(u8) -> T + Sync,
    ) -> Vec<T> {
        let plane = s * s;
        let mut out = vec![T::default(); items.len() * 3 * plane];
        let fill = |(k, out): (usize, &mut [T])| {
            let img = &items[k].sample.image;
            debug_assert_eq!((img.width as usize, img.height as usize), (s, s));
            let (r, rest) = out.split_at_mut(plane);
            let (g, b) = rest.split_at_mut(plane);
            for (i, px) in img.data.as_chunks::<3>().0.iter().enumerate() {
                r[i] = conv(px[0]);
                g[i] = conv(px[1]);
                b[i] = conv(px[2]);
            }
        };
        if par::is_serial() || items.len() == 1 {
            out.chunks_mut(3 * plane).enumerate().for_each(fill);
        } else {
            use rayon::prelude::*;
            out.par_chunks_mut(3 * plane).enumerate().for_each(fill);
        }
        out
    }
    let (images, images_u8) = if cfg.u8_images {
        (Vec::new(), planar(&items, s, |v| v))
    } else {
        let lut: [f32; 256] = std::array::from_fn(|i| i as f32 / 255.0);
        (planar(&items, s, |v| lut[v as usize]), Vec::new())
    };
    debug_assert_eq!(images.len().max(images_u8.len()), n * 3 * plane);
    let mut targets = Vec::new();
    let mut meta = Vec::with_capacity(n);
    for (k, t) in items.iter().enumerate() {
        for b in &t.sample.boxes {
            targets.push([k as f32, b.class as f32, b.cx, b.cy, b.w, b.h]);
        }
        meta.push(ImageMeta {
            index: t.index,
            orig_size: t.orig_size,
            ratio: t.letterbox.ratio,
            pad: t.letterbox.pad,
        });
    }
    Ok(Batch { images, images_u8, shape: [n, 3, s, s], targets, meta })
}

/// Iterator over one epoch's batches. Dropping it stops the workers and joins them.
pub struct EpochIter {
    started: Instant,
    images: usize,
    batches: usize,
    num_batches: usize,
    inner: Inner,
}

impl EpochIter {
    /// Images per second delivered so far.
    pub fn throughput(&self) -> Throughput {
        let seconds = self.started.elapsed().as_secs_f64();
        Throughput {
            images: self.images,
            batches: self.batches,
            seconds,
            images_per_s: if seconds > 0.0 { self.images as f64 / seconds } else { 0.0 },
        }
    }

    /// Batches in this epoch.
    pub fn num_batches(&self) -> usize {
        self.num_batches
    }
}

impl Iterator for EpochIter {
    type Item = Result<Batch>;

    fn next(&mut self) -> Option<Result<Batch>> {
        if self.batches >= self.num_batches {
            return None;
        }
        let batch = match &mut self.inner {
            Inner::Sync { shared, transform, order, epoch } => {
                let bs = shared.cfg.batch_size;
                let (lo, hi) = (self.batches * bs, ((self.batches + 1) * bs).min(order.len()));
                let items = (lo..hi).map(|p| shared.produce(transform, *epoch, p, order[p])).collect();
                collate(items, &shared.cfg)
            }
            Inner::Threads { rx, .. } => match rx.as_ref()?.recv() {
                Ok(b) => b,
                Err(_) => {
                    // The pipeline ended early; report it once instead of silently truncating the epoch.
                    self.batches = self.num_batches;
                    return Some(Err(DataError::Config("data loader threads stopped unexpectedly".into())));
                }
            },
        };
        self.batches += 1;
        if let Ok(b) = &batch {
            self.images += b.len();
        }
        Some(batch)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let left = self.num_batches - self.batches;
        (left, Some(left))
    }
}

impl Drop for EpochIter {
    fn drop(&mut self) {
        if let Inner::Threads { rx, gate, threads } = &mut self.inner {
            gate.shutdown();
            drop(rx.take()); // unblocks the collator's send
            for t in threads.drain(..) {
                let _ = t.join();
            }
        }
    }
}
