//! Data pipeline throughput at 640 on a synthetic JPEG dataset.
//!
//! `cargo bench -p folo-data --bench pipeline [-- --images N]`
//!
//! Prints single-thread stage timings (decode, letterbox, mosaic, affine, HSV) and end-to-end `DataLoader` images/s
//! for the full training transform (decode + mosaic + affine + HSV + flip + collate) with 1 and N workers.

use folo_data::augment::{mosaic4, random_affine, random_hsv, resize_long_side};
use folo_data::synthetic::{SyntheticConfig, write_yolo};
use folo_data::{
    AugmentConfig, DataLoader, Dataset, ImageFormat, LetterboxOptions, LoaderConfig, Rng, Split,
    letterbox_sample, open,
};
use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

const IMGSZ: u32 = 640;

fn per_image(label: &str, n: usize, f: impl FnMut(usize)) {
    let mut f = f;
    // Warm up once.
    f(0);
    let t = Instant::now();
    for i in 0..n {
        f(i);
    }
    let s = t.elapsed().as_secs_f64();
    println!("| {label:<34} | {:>8.2} | {:>9.0} |", s * 1e3 / n as f64, n as f64 / s);
}

fn main() -> folo_data::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let images =
        args.iter().position(|a| a == "--images").and_then(|i| args.get(i + 1)?.parse().ok()).unwrap_or(256);
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get());

    let dir = std::env::temp_dir().join(format!("folo-data-bench-{images}"));
    let cfg = SyntheticConfig { images, imgsz: IMGSZ, seed: 0, val_fraction: 0.0, ..Default::default() };
    let t = Instant::now();
    let _ = std::fs::remove_dir_all(&dir);
    write_yolo(&cfg, &dir, ImageFormat::Jpeg(90))?;
    println!(
        "wrote {images} synthetic {IMGSZ}px JPEGs in {:.2}s ({threads} threads)\n",
        t.elapsed().as_secs_f64()
    );
    let ds = open(&dir)?;
    let train = Arc::new(ds.split(Split::Train).clone());
    let n = train.len();

    println!("Single thread, per image (stages)\n");
    println!("| stage                              | ms/img   | img/s     |");
    println!("|------------------------------------|----------|-----------|");
    let samples: Vec<_> = (0..n.min(64)).map(|i| train.get(i).unwrap()).collect();
    let m = samples.len();
    folo_data::par::serial(|| {
        per_image("decode JPEG (640 long side)", n.min(128), |i| {
            black_box(train.get(i).unwrap());
        });
        per_image("letterbox 640 (resize + pad)", m, |i| {
            black_box(letterbox_sample(&samples[i], IMGSZ, IMGSZ, &LetterboxOptions::default()));
        });
        let resized: Vec<_> = samples.iter().map(|s| resize_long_side(s.clone(), IMGSZ)).collect();
        let mut rng = Rng::new(1);
        per_image("mosaic4 (1280 canvas)", m, |i| {
            let p = [&resized[i], &resized[(i + 1) % m], &resized[(i + 2) % m], &resized[(i + 3) % m]];
            black_box(mosaic4(p, IMGSZ, &mut rng));
        });
        let canvas = mosaic4([&resized[0], &resized[1], &resized[2], &resized[3]], IMGSZ, &mut rng);
        let aug = AugmentConfig { degrees: 10.0, shear: 2.0, ..AugmentConfig::default() };
        per_image("random affine 1280 -> 640", m, |_| {
            black_box(random_affine(&canvas, IMGSZ, IMGSZ, &aug, &mut rng));
        });
        let mut img = letterbox_sample(&samples[0], IMGSZ, IMGSZ, &LetterboxOptions::default()).0.image;
        per_image("HSV jitter 640x640", m, |_| {
            random_hsv(&mut img, [0.015, 0.7, 0.4], &mut rng);
        });
    });

    println!(
        "\nDataLoader, full train transform (decode + mosaic + affine + HSV + flip + collate), batch 16\n"
    );
    println!("| workers | epoch images | seconds | img/s   |");
    println!("|---------|--------------|---------|---------|");
    let mut worker_counts = vec![1, threads / 2, threads];
    worker_counts.dedup();
    for workers in worker_counts.into_iter().filter(|&w| w > 0) {
        let loader = DataLoader::new(
            train.clone(),
            LoaderConfig {
                batch_size: 16,
                imgsz: IMGSZ,
                workers,
                seed: 0,
                prefetch_batches: 4,
                ..LoaderConfig::default()
            },
        )?;
        // Warm-up epoch (page cache, allocator), then a measured one.
        for b in loader.epoch(0) {
            black_box(b?);
        }
        let mut it = loader.epoch(1);
        for b in it.by_ref() {
            black_box(b?);
        }
        let tp = it.throughput();
        println!("| {workers:>7} | {:>12} | {:>7.2} | {:>7.1} |", tp.images, tp.seconds, tp.images_per_s);
    }

    println!("\nDataLoader, validation (decode + letterbox + collate), batch 16\n");
    println!("| workers | img/s   |");
    println!("|---------|---------|");
    for workers in [1, threads] {
        let loader =
            DataLoader::new(train.clone(), LoaderConfig { workers, ..LoaderConfig::val(16, IMGSZ) })?;
        let mut it = loader.epoch(0);
        for b in it.by_ref() {
            black_box(b?);
        }
        println!("| {workers:>7} | {:>7.1} |", it.throughput().images_per_s);
    }
    let _ = std::fs::remove_dir_all(&dir);
    Ok(())
}
