//! Data pipeline throughput: stage timings and `DataLoader` images/s.
//!
//! `cargo bench -p folo-data --bench pipeline [-- OPTIONS]`
//!
//! - `--images N`: size of the synthetic 640 px JPEG dataset written when no `--data` is given (default 256).
//! - `--data PATH [--split train|val]`: a real dataset instead (YOLO or COCO; default split `train`).
//! - `--imgsz S`: network size (default 640).
//! - `--workers A,B,..`: loader worker counts (default `1,8,<cores>`).
//! - `--batches N`: stop each measured epoch after N batches (default: the whole epoch).
//! - `--rounds N`: measured epochs per worker count; the median and the range are printed (default 3).
//! - `--no-stages` / `--no-val`: skip the single-thread stage table / the validation loader table.
//! - `--cache ram`: the training loader with `CacheMode::Ram` (its warm-up epoch is a whole epoch, which fills
//!   the cache; the measured epochs then read from it).
//! - `--mosaic-window K`: the training loader with `AugmentConfig::mosaic_window = K` (default 0: partners from the
//!   whole dataset).
//! - `--epoch E`: the first measured epoch (default 1; the warm-up is epoch 0). Arms of an A/B on a large dataset
//!   take different epochs, so neither reads files the other has just brought into the page cache.
//!
//! Prints single-thread stage timings (decode, resize, letterbox, mosaic, affine, HSV, flip, collate) and
//! end-to-end `DataLoader` images/s for the full training transform (decode + mosaic + affine + HSV + flip +
//! collate to `u8`, as the GPU trainer asks for) and for validation (decode + letterbox + collate).

use folo_data::augment::{flip_lr, mosaic4, random_affine, random_hsv, resize_long_side};
use folo_data::synthetic::{SyntheticConfig, write_yolo};
use folo_data::{
    AugmentConfig, CacheMode, DataLoader, Dataset, ImageFormat, LetterboxOptions, LoaderConfig, Rng, Split,
    letterbox_sample, open,
};
use std::hint::black_box;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

fn per_image(label: &str, n: usize, f: impl FnMut(usize)) {
    let mut f = f;
    // Warm up once.
    f(0);
    let t = Instant::now();
    for i in 0..n {
        f(i);
    }
    let s = t.elapsed().as_secs_f64();
    println!("| {label:<34} | {:>8.3} | {:>9.0} |", s * 1e3 / n as f64, n as f64 / s);
}

struct Args {
    images: usize,
    data: Option<PathBuf>,
    split: Split,
    imgsz: u32,
    workers: Vec<usize>,
    batches: usize,
    rounds: usize,
    stages: bool,
    val: bool,
    cache: CacheMode,
    mosaic_window: u32,
    epoch: u32,
}

fn parse_args() -> Args {
    let argv: Vec<String> = std::env::args().collect();
    let value = |name: &str| argv.iter().position(|a| a == name).and_then(|i| argv.get(i + 1)).cloned();
    let flag = |name: &str| argv.iter().any(|a| a == name);
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get());
    let mut workers: Vec<usize> = match value("--workers") {
        Some(w) => w.split(',').filter_map(|v| v.parse().ok()).collect(),
        None => vec![1, 8.min(threads), threads],
    };
    workers.dedup();
    Args {
        images: value("--images").and_then(|v| v.parse().ok()).unwrap_or(256),
        data: value("--data").map(PathBuf::from),
        split: match value("--split").as_deref() {
            Some("val") => Split::Val,
            Some("test") => Split::Test,
            _ => Split::Train,
        },
        imgsz: value("--imgsz").and_then(|v| v.parse().ok()).unwrap_or(640),
        workers,
        batches: value("--batches").and_then(|v| v.parse().ok()).unwrap_or(usize::MAX),
        rounds: value("--rounds").and_then(|v| v.parse().ok()).unwrap_or(3).max(1),
        stages: !flag("--no-stages"),
        val: !flag("--no-val"),
        cache: if value("--cache").as_deref() == Some("ram") { CacheMode::Ram } else { CacheMode::None },
        mosaic_window: value("--mosaic-window").and_then(|v| v.parse().ok()).unwrap_or(0),
        epoch: value("--epoch").and_then(|v| v.parse().ok()).unwrap_or(1),
    }
}

/// Images per second of `batches` batches of one epoch (the wall time from the epoch's start, thread start-up
/// included, to its last batch).
fn epoch_rate(loader: &DataLoader, epoch: u32, batches: usize) -> folo_data::Result<f64> {
    let t = Instant::now();
    let mut images = 0usize;
    for b in loader.epoch(epoch).take(batches) {
        images += black_box(b?).len();
    }
    Ok(images as f64 / t.elapsed().as_secs_f64())
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn main() -> folo_data::Result<()> {
    let args = parse_args();
    let s = args.imgsz;
    let mut scratch = None;
    let dir = match &args.data {
        Some(p) => p.clone(),
        None => {
            // Off the system drive when the test scratch root is set (AGENTS.md, build hygiene).
            let root = std::env::var_os("FOLO_TEST_TMP").map_or_else(std::env::temp_dir, PathBuf::from);
            let dir = root.join(format!("folo-data-bench-{}-{}", args.images, std::process::id()));
            let cfg = SyntheticConfig {
                images: args.images,
                imgsz: 640,
                seed: 0,
                val_fraction: 0.0,
                ..Default::default()
            };
            let t = Instant::now();
            let _ = std::fs::remove_dir_all(&dir);
            write_yolo(&cfg, &dir, ImageFormat::Jpeg(90))?;
            println!("wrote {} synthetic 640px JPEGs in {:.2}s\n", args.images, t.elapsed().as_secs_f64());
            scratch = Some(dir.clone());
            dir
        }
    };
    let t = Instant::now();
    let ds = open(&dir)?;
    let train = Arc::new(ds.split(args.split).clone());
    let n = train.len();
    println!(
        "{}: {n} images ({} split), opened in {:.2}s; imgsz {s}\n",
        dir.display(),
        args.split,
        t.elapsed().as_secs_f64()
    );

    if args.stages {
        println!("Single thread, per image (stages)\n");
        println!("| stage                              | ms/img   | img/s     |");
        println!("|------------------------------------|----------|-----------|");
        let samples: Vec<_> = (0..n.min(64)).map(|i| train.get(i).unwrap()).collect();
        let m = samples.len();
        let (w, h) = (samples[0].image.width, samples[0].image.height);
        folo_data::par::serial(|| {
            per_image(&format!("decode ({w}x{h} first image)"), n.min(128), |i| {
                black_box(train.get(i).unwrap());
            });
            per_image(&format!("resize long side to {s}"), m, |i| {
                black_box(resize_long_side(samples[i].clone(), s));
            });
            per_image("  (of which: clone the sample)", m, |i| {
                black_box(samples[i].clone());
            });
            per_image(&format!("letterbox {s} (resize + pad)"), m, |i| {
                black_box(letterbox_sample(&samples[i], s, s, &LetterboxOptions::default()));
            });
            let resized: Vec<_> = samples.iter().map(|x| resize_long_side(x.clone(), s)).collect();
            let mut rng = Rng::new(1);
            per_image(&format!("mosaic4 ({} canvas)", 2 * s), m, |i| {
                let p = [&resized[i], &resized[(i + 1) % m], &resized[(i + 2) % m], &resized[(i + 3) % m]];
                black_box(mosaic4(p, s, &mut rng));
            });
            let canvas =
                mosaic4([&resized[0], &resized[1 % m], &resized[2 % m], &resized[3 % m]], s, &mut rng);
            let aug = AugmentConfig::default();
            per_image(&format!("random affine {} -> {s} (default)", 2 * s), m, |_| {
                black_box(random_affine(&canvas, s, s, &aug, &mut rng));
            });
            let rot = AugmentConfig { degrees: 10.0, shear: 2.0, ..AugmentConfig::default() };
            per_image(&format!("random affine {} -> {s} (rotate)", 2 * s), m, |_| {
                black_box(random_affine(&canvas, s, s, &rot, &mut rng));
            });
            let mut lb = letterbox_sample(&samples[0], s, s, &LetterboxOptions::default()).0;
            per_image(&format!("HSV jitter {s}x{s}"), m, |_| {
                random_hsv(&mut lb.image, [0.015, 0.7, 0.4], &mut rng);
            });
            per_image(&format!("flip left-right {s}x{s}"), m, |_| {
                flip_lr(&mut lb);
            });
        });
        println!();
    }

    println!(
        "DataLoader, full train transform (decode + mosaic + affine + HSV + flip + collate to u8), batch 16, \
         mosaic window {}, median of {} epochs from epoch {}\n",
        args.mosaic_window, args.rounds, args.epoch
    );
    println!("| workers | img/s   | range           |");
    println!("|---------|---------|-----------------|");
    for &workers in args.workers.iter().filter(|&&w| w > 0) {
        let loader = DataLoader::new(
            train.clone(),
            LoaderConfig {
                batch_size: 16,
                imgsz: s,
                workers,
                seed: 0,
                u8_images: true,
                cache: args.cache,
                augment: Some(AugmentConfig { mosaic_window: args.mosaic_window, ..AugmentConfig::default() }),
                ..LoaderConfig::default()
            },
        )?;
        // Warm-up (page cache, allocator; a whole epoch fills a RAM cache), then the measured epochs.
        let warm = if args.cache == CacheMode::Ram { usize::MAX } else { args.batches.min(8) };
        epoch_rate(&loader, 0, warm)?;
        let mut rates = Vec::new();
        for e in 0..args.rounds {
            rates.push(epoch_rate(&loader, args.epoch + e as u32, args.batches)?);
        }
        let med = median(&mut rates);
        println!("| {workers:>7} | {med:>7.1} | {:>6.1} - {:>6.1} |", rates[0], rates[rates.len() - 1]);
    }

    if args.val {
        println!("\nDataLoader, validation (decode + letterbox + collate to u8), batch 32\n");
        println!("| workers | img/s   | range           |");
        println!("|---------|---------|-----------------|");
        for &workers in args.workers.iter().filter(|&&w| w > 0) {
            let loader = DataLoader::new(
                train.clone(),
                LoaderConfig { workers, u8_images: true, ..LoaderConfig::val(32, s) },
            )?;
            epoch_rate(&loader, 0, args.batches.min(4))?;
            let mut rates = Vec::new();
            for _ in 0..args.rounds {
                rates.push(epoch_rate(&loader, 0, args.batches)?);
            }
            let med = median(&mut rates);
            println!("| {workers:>7} | {med:>7.1} | {:>6.1} - {:>6.1} |", rates[0], rates[rates.len() - 1]);
        }
    }
    if let Some(dir) = scratch {
        let _ = std::fs::remove_dir_all(&dir);
    }
    Ok(())
}
