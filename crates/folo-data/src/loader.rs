//! Multi-threaded, deterministic batching data loader.
//!
//! Each epoch has a fixed sample order (a seeded shuffle of `(seed, epoch)`) and every sample position has its own
//! RNG stream `(seed, epoch, position)`. Worker threads claim positions from a shared counter, decode and augment,
//! and send results to a collator thread that restores the order and builds [`Batch`]es. The output is therefore
//! identical for any number of workers. In-flight work is bounded by a window of positions plus a bounded batch
//! channel; dropping the epoch iterator stops and joins all threads.

use crate::augment::{AugmentConfig, FetchedSample, ProducedMeta, TrainTransform};
use crate::buffers;
use crate::dataset::{Dataset, Issue, IssueKind};
use crate::error::{DataError, Result};
use crate::par;
use crate::ram_cache::{CacheMode, RamCache};
use crate::rng::Rng;
use crate::simd::PlanarDst;
use crossbeam_channel::{Receiver, bounded, unbounded};
use std::collections::BTreeMap;
use std::mem::ManuallyDrop;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
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
    fn fetch(&self, i: usize) -> Result<FetchedSample> {
        let n = self.ds.len();
        let mut last = None;
        for k in 0..MAX_SUBSTITUTES.min(n) {
            let j = (i + k) % n;
            let got = match &self.ram {
                Some(c) => c.get(j, || self.ds.get(j)).map(|entry| FetchedSample::Shared { index: j, entry }),
                None => self.ds.get(j).map(|s| FetchedSample::decoded(j, s)),
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

    /// Transforms the sample at epoch position `pos` straight into `dst`. A panic (a bug in a dataset or transform)
    /// becomes an error for that batch instead of silently ending the epoch.
    fn produce_into(
        &self,
        transform: &TrainTransform,
        epoch: u32,
        pos: usize,
        index: usize,
        dst: &mut PlanarDst<'_>,
    ) -> Result<ProducedMeta> {
        let mut rng = Rng::derive(&[self.cfg.seed, epoch as u64, pos as u64]);
        let run = || transform.apply_into_planar(index, self.ds.len(), &|i| self.fetch(i), &mut rng, dst);
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
        let elem_bytes = if cfg.u8_images { size_of::<u8>() } else { size_of::<f32>() };
        let batch_bytes = (cfg.imgsz as usize)
            .checked_mul(cfg.imgsz as usize)
            .and_then(|p| p.checked_mul(3))
            .and_then(|slot| slot.checked_mul(cfg.batch_size))
            .and_then(|elems| elems.checked_mul(elem_bytes))
            .filter(|&b| b <= isize::MAX as usize);
        if batch_bytes.is_none() {
            return Err(DataError::Config(format!(
                "batch_size {} at imgsz {} overflows batch buffer size",
                cfg.batch_size, cfg.imgsz
            )));
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

/// Owning raw allocation for a batch's planar image buffer (`Vec<u8>` or `Vec<f32>`), split by raw pointer
/// provenance so worker threads can write strictly disjoint image slots with zero overlapping `&mut Vec` references
/// and zero collator copy.
enum RawBatchBuf {
    U8 { ptr: *mut u8, len: usize, cap: usize },
    F32 { ptr: *mut f32, len: usize, cap: usize },
}

impl RawBatchBuf {
    fn new(total_elems: usize, u8_images: bool) -> Self {
        if u8_images {
            let mut v = ManuallyDrop::new(vec![0u8; total_elems]);
            RawBatchBuf::U8 { ptr: v.as_mut_ptr(), len: v.len(), cap: v.capacity() }
        } else {
            let mut v = ManuallyDrop::new(vec![0.0f32; total_elems]);
            RawBatchBuf::F32 { ptr: v.as_mut_ptr(), len: v.len(), cap: v.capacity() }
        }
    }

    fn into_vecs(self) -> (Vec<f32>, Vec<u8>) {
        let this = ManuallyDrop::new(self);
        match *this {
            // SAFETY: `(ptr, len, cap)` came from a live `Vec` of the matching type and `this` will not run `Drop`.
            RawBatchBuf::U8 { ptr, len, cap } => (Vec::new(), unsafe { Vec::from_raw_parts(ptr, len, cap) }),
            RawBatchBuf::F32 { ptr, len, cap } => (unsafe { Vec::from_raw_parts(ptr, len, cap) }, Vec::new()),
        }
    }
}

impl Drop for RawBatchBuf {
    fn drop(&mut self) {
        match *self {
            // SAFETY: reconstructs the exact `Vec` disassembled in `RawBatchBuf::new` once no `Slot` remains.
            RawBatchBuf::U8 { ptr, len, cap } => drop(unsafe { Vec::from_raw_parts(ptr, len, cap) }),
            RawBatchBuf::F32 { ptr, len, cap } => drop(unsafe { Vec::from_raw_parts(ptr, len, cap) }),
        }
    }
}

struct BatchAlloc {
    buf: RawBatchBuf,
    slot_elems: usize,
    plane: usize,
}

// SAFETY: `BatchAlloc` never reallocates; every `Slot` receives a unique `slot_idx` from `BatchEntry::take_slot`
// and creates `&mut [T]` strictly over `[slot_idx * slot_elems .. (slot_idx + 1) * slot_elems)`. `into_vecs` and
// `Drop` run only when the `Arc<BatchAlloc>` refcount drops to 1 / 0 (zero remaining `Slot`s).
unsafe impl Send for BatchAlloc {}
unsafe impl Sync for BatchAlloc {}

/// Exclusive write handle to one image's planar `[3, s, s]` slice inside a shared [`BatchAlloc`].
struct Slot {
    alloc: Arc<BatchAlloc>,
    slot_idx: usize,
}

impl Slot {
    fn planar_dst(&mut self) -> PlanarDst<'_> {
        let start = self.slot_idx.checked_mul(self.alloc.slot_elems).expect("slot offset overflow");
        let len = self.alloc.slot_elems;
        let end = start.checked_add(len).expect("slot end overflow");
        let plane = self.alloc.plane;
        match self.alloc.buf {
            RawBatchBuf::U8 { ptr, len: total, .. } => {
                assert!(end <= total, "batch u8 slot {start}..{end} exceeds {total}");
                // SAFETY: `start..end` is bounds-checked and exclusively owned by this `Slot` (`&mut self` prevents
                // self-aliasing).
                PlanarDst::from_u8(unsafe { std::slice::from_raw_parts_mut(ptr.add(start), len) }, plane)
            }
            RawBatchBuf::F32 { ptr, len: total, .. } => {
                assert!(end <= total, "batch f32 slot {start}..{end} exceeds {total}");
                // SAFETY: as above.
                PlanarDst::from_f32_div255(
                    unsafe { std::slice::from_raw_parts_mut(ptr.add(start), len) },
                    plane,
                )
            }
        }
    }
}

impl BatchAlloc {
    fn new(len: usize, imgsz: usize, u8_images: bool) -> Self {
        let plane = imgsz.checked_mul(imgsz).expect("imgsz overflow");
        let slot_elems = plane.checked_mul(3).expect("slot size overflow");
        let total_elems = len.checked_mul(slot_elems).expect("batch size overflow");
        let elem_bytes = if u8_images { size_of::<u8>() } else { size_of::<f32>() };
        assert!(
            total_elems.checked_mul(elem_bytes).is_some_and(|b| b <= isize::MAX as usize),
            "batch buffer exceeds isize::MAX bytes"
        );
        let buf = RawBatchBuf::new(total_elems, u8_images);
        BatchAlloc { buf, slot_elems, plane }
    }
}

struct BatchEntry {
    /// Lazily initialized outside the [`Gate`] mutex so allocating/zero-faulting a multi-megabyte batch buffer
    /// never blocks other batches, workers, or the collator's [`Gate::advance`].
    alloc: Arc<OnceLock<Arc<BatchAlloc>>>,
    next_slot: usize,
    len: usize,
}

impl BatchEntry {
    fn new(len: usize) -> Self {
        BatchEntry { alloc: Arc::new(OnceLock::new()), next_slot: 0, len }
    }

    fn take_slot(&mut self) -> (Arc<OnceLock<Arc<BatchAlloc>>>, usize, bool) {
        let idx = self.next_slot;
        assert!(idx < self.len, "batch slot issued past batch length");
        self.next_slot += 1;
        (self.alloc.clone(), idx, self.next_slot == self.len)
    }
}

/// Window gate: workers may claim position `p` only while `p < done + window`, receiving that position's exclusive
/// write [`Slot`] in its batch buffer.
struct Gate {
    state: Mutex<GateState>,
    cv: Condvar,
    stop: AtomicBool,
    batch_size: usize,
    imgsz: usize,
    u8_images: bool,
}

struct GateState {
    next: usize,
    done: usize,
    /// Active batch currently being handed out; positions are issued strictly monotonically, so at most one
    /// batch is ever partially claimed at a time.
    current_batch: Option<BatchEntry>,
}

impl Gate {
    fn claim(&self, total: usize, window: usize) -> Option<(usize, Slot)> {
        let (pos, batch_len, cell, slot_idx) = {
            let mut st = self.state.lock().unwrap_or_else(|p| p.into_inner());
            loop {
                if self.stop.load(Ordering::Acquire) || st.next >= total {
                    return None;
                }
                if st.next < st.done.saturating_add(window) {
                    let pos = st.next;
                    st.next += 1;
                    let b = pos / self.batch_size;
                    let lo = b * self.batch_size;
                    let batch_len = (lo + self.batch_size).min(total) - lo;
                    let entry = st.current_batch.get_or_insert_with(|| BatchEntry::new(batch_len));
                    let (cell, slot_idx, last) = entry.take_slot();
                    if last {
                        st.current_batch = None;
                    }
                    break (pos, batch_len, cell, slot_idx);
                }
                st = self.cv.wait(st).unwrap_or_else(|p| p.into_inner());
            }
        };
        if self.stop.load(Ordering::Acquire) {
            return None;
        }
        // Allocate and zero-fault the batch buffer outside `self.state` so unrelated batches and `advance` proceed
        // concurrently. Dropping `cell` before returning guarantees `OnceLock` holds no extra `Arc<BatchAlloc>` ref
        // once all slots of the batch have been returned and finished.
        let alloc =
            cell.get_or_init(|| Arc::new(BatchAlloc::new(batch_len, self.imgsz, self.u8_images))).clone();
        drop(cell);
        Some((pos, Slot { alloc, slot_idx }))
    }

    fn advance(&self, done: usize) {
        self.state.lock().unwrap_or_else(|p| p.into_inner()).done = done;
        self.cv.notify_all();
    }

    fn shutdown(&self) {
        self.stop.store(true, Ordering::Release);
        let mut st = self.state.lock().unwrap_or_else(|p| p.into_inner());
        st.current_batch = None;
        self.cv.notify_all();
    }
}

struct DoneSample {
    meta: ProducedMeta,
    slot: Slot,
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
    let window = cfg
        .batch_size
        .saturating_mul(cfg.prefetch_batches.saturating_add(1))
        .saturating_add(cfg.workers)
        .min(total.max(1))
        .max(1);
    let ring_cap = window.next_power_of_two();
    let ring_mask = ring_cap - 1;
    let gate = Arc::new(Gate {
        state: Mutex::new(GateState { next: 0, done: 0, current_batch: None }),
        cv: Condvar::new(),
        stop: false.into(),
        batch_size: cfg.batch_size,
        imgsz: cfg.imgsz as usize,
        u8_images: cfg.u8_images,
    });
    let (sample_tx, sample_rx) = unbounded::<(usize, Result<DoneSample>)>();
    let (batch_tx, batch_rx) = bounded::<Result<Batch>>(cfg.prefetch_batches.max(1));
    let mut threads = Vec::with_capacity(cfg.workers + 1);
    for w in 0..cfg.workers {
        let (shared, order, gate, tx) = (shared.clone(), order.clone(), gate.clone(), sample_tx.clone());
        let handle = std::thread::Builder::new()
            .name(format!("folo-data-{w}"))
            .spawn(move || {
                buffers::pooled(|| {
                    par::serial(|| {
                        while let Some((pos, mut slot)) = gate.claim(total, window) {
                            let out = shared
                                .produce_into(&transform, epoch, pos, order[pos], &mut slot.planar_dst())
                                .map(|meta| DoneSample { meta, slot });
                            if tx.send((pos, out)).is_err() {
                                break;
                            }
                        }
                    })
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
                buffers::pooled(|| {
                    let bs = shared.cfg.batch_size;
                    let s = shared.cfg.imgsz as usize;
                    // Every in-flight sample satisfies `lo <= p < lo + window` and `ring_cap >= window`, so
                    // `p & ring_mask` gives each active position a unique O(1) slot with zero hashing.
                    let mut pending: Vec<Option<Result<DoneSample>>> = (0..ring_cap).map(|_| None).collect();
                    let mut batch_buf: Vec<DoneSample> = Vec::with_capacity(bs);
                    for b in 0..num_batches {
                        let (lo, hi) = (b * bs, ((b + 1) * bs).min(total));
                        let need = hi - lo;
                        let mut ready = 0usize;
                        for p in lo..hi {
                            if pending[p & ring_mask].is_some() {
                                ready += 1;
                            }
                        }
                        while ready < need {
                            match sample_rx.recv() {
                                Ok((p, sample)) => {
                                    debug_assert!(
                                        p >= lo && p < lo.saturating_add(window),
                                        "sample position {p} outside active window {lo}..{}",
                                        lo.saturating_add(window)
                                    );
                                    let slot = &mut pending[p & ring_mask];
                                    debug_assert!(slot.is_none(), "duplicate sample position {p}");
                                    *slot = Some(sample);
                                    if p >= lo && p < hi {
                                        ready += 1;
                                    }
                                }
                                Err(_) => return, // workers stopped
                            }
                        }
                        // Drain every slot in `lo..hi` unconditionally before advancing the gate so no `Slot`
                        // remains in `pending` even if an item in this batch failed.
                        batch_buf.clear();
                        let mut first_err: Option<DataError> = None;
                        for p in lo..hi {
                            match pending[p & ring_mask].take().expect("batch slot ready") {
                                Ok(item) => batch_buf.push(item),
                                Err(e) => {
                                    if first_err.is_none() {
                                        first_err = Some(e);
                                    }
                                }
                            }
                        }
                        let batch = match first_err {
                            Some(e) => {
                                batch_buf.clear();
                                Err(e)
                            }
                            None => Ok(finish_batch(&mut batch_buf, s)),
                        };
                        gate.advance(hi);
                        if batch_tx.send(batch).is_err() {
                            return;
                        }
                    }
                })
            })
            .expect("failed to spawn the data loader collator")
    };
    threads.push(collator);
    Inner::Threads { rx: Some(batch_rx), gate, threads }
}

/// Assembles a [`Batch`] from completed [`DoneSample`]s without copying pixel data.
fn finish_batch(items: &mut Vec<DoneSample>, s: usize) -> Batch {
    let n = items.len();
    let total_boxes: usize = items.iter().map(|item| item.meta.boxes.len()).sum();
    let mut targets = Vec::with_capacity(total_boxes);
    let mut meta = Vec::with_capacity(n);
    let mut last_alloc: Option<Arc<BatchAlloc>> = None;
    for (k, item) in items.drain(..).enumerate() {
        for b in &item.meta.boxes {
            targets.push([k as f32, b.class as f32, b.cx, b.cy, b.w, b.h]);
        }
        meta.push(ImageMeta {
            index: item.meta.index,
            orig_size: item.meta.orig_size,
            ratio: item.meta.letterbox.ratio,
            pad: item.meta.letterbox.pad,
        });
        last_alloc = Some(item.slot.alloc);
    }
    let alloc = Arc::into_inner(last_alloc.expect("non-empty batch"))
        .expect("all batch slots finished before finish_batch");
    let (images, images_u8) = alloc.buf.into_vecs();
    debug_assert_eq!(images.len().max(images_u8.len()), n * 3 * s * s);
    Batch { images, images_u8, shape: [n, 3, s, s], targets, meta }
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
                let s = shared.cfg.imgsz as usize;
                let plane = s * s;
                let (lo, hi) = (self.batches * bs, ((self.batches + 1) * bs).min(order.len()));
                let n = hi - lo;
                let mut images = if shared.cfg.u8_images { Vec::new() } else { vec![0.0f32; n * 3 * plane] };
                let mut images_u8 = if shared.cfg.u8_images { vec![0u8; n * 3 * plane] } else { Vec::new() };
                let mut targets = Vec::new();
                let mut meta = Vec::with_capacity(n);
                let mut err = None;
                for (k, p) in (lo..hi).enumerate() {
                    let mut dst = if shared.cfg.u8_images {
                        PlanarDst::from_u8(&mut images_u8[k * 3 * plane..(k + 1) * 3 * plane], plane)
                    } else {
                        PlanarDst::from_f32_div255(&mut images[k * 3 * plane..(k + 1) * 3 * plane], plane)
                    };
                    match shared.produce_into(transform, *epoch, p, order[p], &mut dst) {
                        Ok(m) => {
                            for b in &m.boxes {
                                targets.push([k as f32, b.class as f32, b.cx, b.cy, b.w, b.h]);
                            }
                            meta.push(ImageMeta {
                                index: m.index,
                                orig_size: m.orig_size,
                                ratio: m.letterbox.ratio,
                                pad: m.letterbox.pad,
                            });
                        }
                        Err(e) => {
                            if err.is_none() {
                                err = Some(e);
                            }
                        }
                    }
                }
                match err {
                    Some(e) => Err(e),
                    None => Ok(Batch { images, images_u8, shape: [n, 3, s, s], targets, meta }),
                }
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
