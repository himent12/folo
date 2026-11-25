//! DataLoader: determinism across thread counts, shuffling, drop_last, shutdown, failure substitution.

mod common;

use common::*;
use folo_data::synthetic::{SyntheticConfig, SyntheticDataset};
use folo_data::{
    AugmentConfig, BBox, Batch, CacheMode, DataError, DataLoader, Dataset, InMemoryDataset, LoaderConfig,
    Result, Sample, Split,
};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

fn rect_dataset(n: usize) -> Arc<dyn Dataset> {
    let samples = (0..n).map(|k| random_rect_sample(k as u64, 90 + (k as u32 % 5) * 13, 70, 4)).collect();
    Arc::new(InMemoryDataset {
        samples,
        classes: vec!["a".into(), "b".into(), "c".into(), "d".into()],
        split: Split::Train,
    })
}

fn cfg(workers: usize) -> LoaderConfig {
    LoaderConfig {
        batch_size: 4,
        imgsz: 64,
        workers,
        seed: 3,
        augment: Some(AugmentConfig { degrees: 5.0, mixup: 0.3, ..AugmentConfig::default() }),
        ..LoaderConfig::default()
    }
}

fn collect(loader: &DataLoader, epoch: u32) -> Vec<Batch> {
    loader.epoch(epoch).map(|b| b.unwrap()).collect()
}

#[test]
fn deterministic_for_any_worker_count() {
    let ds = rect_dataset(23);
    let reference = collect(&DataLoader::new(ds.clone(), cfg(0)).unwrap(), 0);
    assert_eq!(reference.len(), 6);
    assert_eq!(reference.last().unwrap().len(), 3);
    for workers in [1, 2, 5] {
        let loader = DataLoader::new(ds.clone(), cfg(workers)).unwrap();
        assert_eq!(collect(&loader, 0), reference, "{workers} workers");
    }
    // Another epoch shuffles and augments differently; the same epoch repeats exactly.
    let loader = DataLoader::new(ds.clone(), cfg(3)).unwrap();
    let e1 = collect(&loader, 1);
    assert_ne!(e1, reference);
    assert_eq!(collect(&loader, 1), e1);
    let mut seen: Vec<usize> = e1.iter().flat_map(|b| b.meta.iter().map(|m| m.index)).collect();
    seen.sort();
    assert_eq!(seen, (0..23).collect::<Vec<_>>(), "each sample exactly once per epoch");
}

#[test]
fn batch_layout() {
    let ds = rect_dataset(8);
    let loader = DataLoader::new(ds, LoaderConfig { augment: None, shuffle: false, ..cfg(2) }).unwrap();
    let b = loader.epoch(0).next().unwrap().unwrap();
    assert_eq!(b.shape, [4, 3, 64, 64]);
    assert_eq!(b.images.len(), 4 * 3 * 64 * 64);
    assert!(b.images.iter().all(|v| (0.0..=1.0).contains(v)));
    assert!(
        b.targets
            .iter()
            .all(|t| (0.0..4.0).contains(&t[0]) && t[2..].iter().all(|v| (0.0..=1.0).contains(v)))
    );
    // Validation geometry maps back to the original image and matches the source boxes.
    for (k, m) in b.meta.iter().enumerate() {
        assert_eq!(m.index, k);
        let src = random_rect_sample(k as u64, 90 + (k as u32 % 5) * 13, 70, 4);
        assert_eq!(m.orig_size, [src.image.width, src.image.height]);
        let rows: Vec<_> = b.targets.iter().filter(|t| t[0] as usize == k).collect();
        assert_eq!(rows.len(), src.boxes.len());
        for (t, sb) in rows.iter().zip(&src.boxes) {
            let net = BBox::new(t[1] as u32, t[2], t[3], t[4], t[5]).pixel_xyxy(64, 64);
            let back = m.unmap_xyxy(net);
            let want = sb.pixel_xyxy(src.image.width, src.image.height);
            for (a, w) in back.iter().zip(want) {
                assert!((a - w).abs() < 1e-3, "{back:?} vs {want:?}");
            }
        }
    }
    // Planes are R, G, B: check one pixel against the letterboxed source.
    let px = |c: usize, y: usize, x: usize| b.images[c * 64 * 64 + y * 64 + x];
    let corner = [px(0, 0, 0), px(1, 0, 0), px(2, 0, 0)];
    assert!(corner.iter().all(|v| (v - 114.0 / 255.0).abs() < 1e-6), "padding is grey 114");
}

#[test]
fn drop_last_and_no_shuffle() {
    let ds = rect_dataset(10);
    let loader =
        DataLoader::new(ds.clone(), LoaderConfig { drop_last: true, shuffle: false, ..cfg(2) }).unwrap();
    assert_eq!(loader.batches_per_epoch(), 2);
    let batches = collect(&loader, 0);
    assert_eq!(batches.len(), 2);
    let order: Vec<usize> = batches.iter().flat_map(|b| b.meta.iter().map(|m| m.index)).collect();
    assert_eq!(order, (0..8).collect::<Vec<_>>());
    let loader = DataLoader::new(ds, LoaderConfig { drop_last: false, ..cfg(2) }).unwrap();
    assert_eq!(loader.batches_per_epoch(), 3);
}

#[test]
fn early_drop_shuts_down() {
    let ds = Arc::new(
        SyntheticDataset::new(SyntheticConfig { images: 200, imgsz: 64, ..Default::default() }, Split::Train)
            .unwrap(),
    );
    let loader = DataLoader::new(ds, LoaderConfig { batch_size: 2, ..cfg(4) }).unwrap();
    for _ in 0..3 {
        let mut it = loader.epoch(0);
        assert!(it.next().unwrap().is_ok());
        let t = it.throughput();
        assert_eq!(t.batches, 1);
        assert!(t.images_per_s > 0.0);
        drop(it); // must not hang
    }
}

/// Wraps a dataset and fails for some indices.
struct Flaky {
    inner: Arc<dyn Dataset>,
    bad: Vec<usize>,
}

impl Dataset for Flaky {
    fn len(&self) -> usize {
        self.inner.len()
    }
    fn get(&self, index: usize) -> Result<Sample> {
        if self.bad.contains(&index) {
            return Err(DataError::Format(format!("sample {index} is broken")));
        }
        self.inner.get(index)
    }
    fn boxes(&self, index: usize) -> &[BBox] {
        self.inner.boxes(index)
    }
    fn class_names(&self) -> &[String] {
        self.inner.class_names()
    }
    fn split(&self) -> Split {
        Split::Train
    }
}

#[test]
fn unreadable_samples_are_substituted_deterministically() {
    let ds: Arc<dyn Dataset> = Arc::new(Flaky { inner: rect_dataset(12), bad: vec![3, 4, 9] });
    let a = DataLoader::new(ds.clone(), cfg(0)).unwrap();
    let ba = collect(&a, 0);
    assert_eq!(ba.iter().map(Batch::len).sum::<usize>(), 12);
    let b = DataLoader::new(ds.clone(), cfg(4)).unwrap();
    assert_eq!(collect(&b, 0), ba);
    let failed: Vec<String> = a.failures().iter().map(|i| i.path.display().to_string()).collect();
    assert_eq!(failed, ["#3", "#4", "#9"]);
    // Metadata names the image that was actually used.
    let v = DataLoader::new(ds.clone(), LoaderConfig { augment: None, shuffle: false, ..cfg(3) }).unwrap();
    let used: Vec<usize> = collect(&v, 0).iter().flat_map(|b| b.meta.iter().map(|m| m.index)).collect();
    assert_eq!(used, [0, 1, 2, 5, 5, 5, 6, 7, 8, 10, 10, 11]);
    let batches = collect(&v, 0);
    let m = batches[0].meta[3];
    assert_eq!(m.orig_size, [90, 70], "size of sample 5 (sample 3 is 129 wide)");
    // Everything broken: the batch is an error, not a panic or a hang.
    let all_bad: Arc<dyn Dataset> = Arc::new(Flaky { inner: rect_dataset(3), bad: vec![0, 1, 2] });
    let l = DataLoader::new(all_bad, LoaderConfig { augment: None, ..cfg(2) }).unwrap();
    assert!(l.epoch(0).next().unwrap().is_err());
    // Sync path (`workers: 0`) must process every position in a failing batch so `failures()` records the same
    // unreadable images as the threaded path (here pos 0 exhausts 0..64, while pos 1..3 hit 64, 65, 66).
    let long_bad: Arc<dyn Dataset> = Arc::new(Flaky { inner: rect_dataset(68), bad: (0..67).collect() });
    let sync_l =
        DataLoader::new(long_bad.clone(), LoaderConfig { augment: None, shuffle: false, ..cfg(0) }).unwrap();
    let par_l = DataLoader::new(long_bad, LoaderConfig { augment: None, shuffle: false, ..cfg(2) }).unwrap();
    assert!(sync_l.epoch(0).next().unwrap().is_err());
    assert!(par_l.epoch(0).next().unwrap().is_err());
    assert_eq!(sync_l.failures().len(), 67);
    assert_eq!(sync_l.failures(), par_l.failures());
}

#[test]
fn close_mosaic_turns_mosaic_off() {
    let ds = rect_dataset(8);
    let c = LoaderConfig {
        epochs: 10,
        augment: Some(AugmentConfig { close_mosaic: 2, ..AugmentConfig::default() }),
        ..cfg(2)
    };
    let loader = DataLoader::new(ds, c).unwrap();
    // Mosaic outputs report an identity letterbox; plain samples report the real letterbox of their image.
    let is_mosaic = |b: &Batch| b.meta.iter().all(|m| m.ratio == [1.0, 1.0]);
    assert!(collect(&loader, 7).iter().all(is_mosaic));
    assert!(collect(&loader, 8).iter().all(|b| !is_mosaic(b)));
}

#[test]
fn invalid_configs_are_rejected() {
    let ds = rect_dataset(2);
    assert!(DataLoader::new(ds.clone(), LoaderConfig { batch_size: 0, ..cfg(1) }).is_err());
    assert!(DataLoader::new(ds.clone(), LoaderConfig { imgsz: 0, ..cfg(1) }).is_err());
    assert!(DataLoader::new(ds.clone(), LoaderConfig { batch_size: usize::MAX, ..cfg(1) }).is_err());
    assert!(DataLoader::new(ds.clone(), LoaderConfig { imgsz: u32::MAX, batch_size: 4, ..cfg(1) }).is_err());
    let bad = AugmentConfig { fliplr: 1.5, ..AugmentConfig::default() };
    assert!(DataLoader::new(ds, LoaderConfig { augment: Some(bad), ..cfg(1) }).is_err());
    let empty: Arc<dyn Dataset> =
        Arc::new(InMemoryDataset { samples: vec![], classes: vec![], split: Split::Val });
    assert!(DataLoader::new(empty, cfg(1)).is_err());
}

/// `--cache ram` gives the batches of the uncached loader, pixel for pixel, box for box and with the same meta
/// (original size, letterbox ratio and pad), for mosaic + mixup, plain letterboxed samples and validation, with
/// images larger and smaller than `imgsz`, from the first (filling) epoch on.
#[test]
fn ram_cache_gives_the_uncached_batches() {
    let samples = (0..11)
        .map(|k| {
            let (w, h) = [(200, 120), (50, 90), (64, 64), (130, 31), (33, 200)][k % 5];
            random_rect_sample(k as u64, w, h, 4)
        })
        .collect();
    let ds: Arc<dyn Dataset> = Arc::new(InMemoryDataset {
        samples,
        classes: (0..4).map(|c| c.to_string()).collect(),
        split: Split::Train,
    });
    let configs = [
        cfg(3),
        LoaderConfig { imgsz: 192, u8_images: true, ..cfg(3) },
        LoaderConfig { augment: Some(AugmentConfig { mosaic: 0.0, ..AugmentConfig::default() }), ..cfg(2) },
        LoaderConfig {
            epochs: 0,
            augment: Some(AugmentConfig { mosaic: 0.0, mixup: 1.0, ..AugmentConfig::default() }),
            ..cfg(2)
        },
        LoaderConfig { augment: None, shuffle: false, ..cfg(2) },
    ];
    for c in configs {
        let plain = DataLoader::new(ds.clone(), c.clone()).unwrap();
        let cached =
            DataLoader::new(ds.clone(), LoaderConfig { cache: CacheMode::Ram, ..c.clone() }).unwrap();
        for epoch in [0, 1, 1] {
            let (a, b) = (collect(&plain, epoch), collect(&cached, epoch));
            assert_eq!(a, b, "{:?}, epoch {epoch}", c.augment.map(|a| a.mosaic));
        }
        if c.augment.is_none() {
            let meta = &collect(&cached, 0)[0].meta[0];
            assert_eq!(meta.orig_size, [200, 120]);
            assert_eq!(meta.ratio, [64.0 / 200.0, 38.0 / 120.0]);
        }
    }
}

/// Wraps a dataset and counts its loads.
struct Counting {
    inner: Arc<dyn Dataset>,
    gets: AtomicUsize,
}

impl Dataset for Counting {
    fn len(&self) -> usize {
        self.inner.len()
    }
    fn get(&self, index: usize) -> Result<Sample> {
        self.gets.fetch_add(1, Ordering::SeqCst);
        self.inner.get(index)
    }
    fn boxes(&self, index: usize) -> &[BBox] {
        self.inner.boxes(index)
    }
    fn class_names(&self) -> &[String] {
        self.inner.class_names()
    }
    fn split(&self) -> Split {
        Split::Train
    }
}

/// `--cache ram` loads each image once: after the first epoch, another loads nothing from the dataset, while the
/// uncached loader loads every image again.
#[test]
fn ram_cache_loads_each_image_once() {
    for cache in [CacheMode::Ram, CacheMode::None] {
        let ds = Arc::new(Counting { inner: rect_dataset(10), gets: AtomicUsize::new(0) });
        let loader = DataLoader::new(ds.clone(), LoaderConfig { cache, ..cfg(3) }).unwrap();
        collect(&loader, 0);
        assert!(ds.gets.swap(0, Ordering::SeqCst) >= 10, "{cache:?}: every image loaded in the first epoch");
        collect(&loader, 1);
        let again = ds.gets.load(Ordering::SeqCst);
        match cache {
            CacheMode::Ram => assert_eq!(again, 0, "the second epoch comes from the cache"),
            CacheMode::None => assert!(again >= 10, "{again} loads in the second epoch"),
        }
    }
}

#[test]
fn ram_cache_refuses_what_does_not_fit() {
    let big = (0..3).map(|k| random_rect_sample(k, 64, 64, 1)).collect();
    let ds: Arc<dyn Dataset> =
        Arc::new(InMemoryDataset { samples: big, classes: vec!["a".into()], split: Split::Train });
    // A 2^20-pixel side: each image needs 3 TiB resized.
    let c = LoaderConfig { imgsz: 1 << 20, cache: CacheMode::Ram, ..cfg(1) };
    let e = DataLoader::new(ds, c).err().expect("refused");
    assert!(e.to_string().contains("--cache ram needs"), "{e}");
}
