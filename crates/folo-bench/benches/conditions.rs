//! Every condition at severity 3 on a 1280 x 720 image. Run with `cargo bench -p folo-bench --bench conditions`
//! (or build with `--release --benches` and run the binary). `FOLO_BENCH_IMAGE=<file>` uses that image (resized
//! to 1280 x 720) instead of the synthetic scene.
//!
//! Prints the median and minimum milliseconds of `apply`, and of `render` (decode + apply + the canonical JPEG
//! encode) for a JPEG of the same image.

use folo_api::BBox;
use folo_bench::conditions::{self, Condition};
use folo_core::rng::Rng;
use image::RgbImage;
use std::hint::black_box;
use std::time::{Duration, Instant};

fn bench<T>(name: &str, iters: usize, mut f: impl FnMut() -> T) {
    black_box(f()); // warm-up
    let mut times: Vec<Duration> = (0..iters)
        .map(|_| {
            let t = Instant::now();
            black_box(f());
            t.elapsed()
        })
        .collect();
    times.sort();
    let (med, min) = (times[times.len() / 2], times[0]);
    println!("{name:<28} median {:>8.2} ms   min {:>8.2} ms   ({iters} iters)", ms(med), ms(min));
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

/// A photo-like scene: smooth gradients, textured noise and a few flat objects.
fn scene(w: u32, h: u32) -> RgbImage {
    let mut rng = Rng::seed(0xC0DE);
    let mut img = RgbImage::from_fn(w, h, |x, y| {
        let (fx, fy) = (x as f32 / w as f32, y as f32 / h as f32);
        image::Rgb([(30.0 + 180.0 * fx) as u8, (50.0 + 150.0 * fy) as u8, (120.0 + 80.0 * (fx - fy)) as u8])
    });
    for p in img.pixels_mut() {
        for c in p.0.iter_mut() {
            *c = (*c as f32 + rng.uniform(-12.0, 12.0)).clamp(0.0, 255.0) as u8;
        }
    }
    for i in 0..12u32 {
        let (x0, y0) = ((i * 97) % (w - 120), (i * 53) % (h - 90));
        let col = [(i * 40 % 255) as u8, (i * 90 % 255) as u8, (i * 150 % 255) as u8];
        for y in y0..y0 + 80 {
            for x in x0..x0 + 110 {
                img.put_pixel(x, y, image::Rgb(col));
            }
        }
    }
    img
}

fn main() {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    println!("folo-bench conditions bench, 1280 x 720, severity 3, {threads} threads available");
    let img = match std::env::var("FOLO_BENCH_IMAGE") {
        Ok(path) => {
            let bytes = std::fs::read(&path).expect("FOLO_BENCH_IMAGE");
            let img = conditions::decode(&bytes).expect("decodable image");
            image::imageops::resize(&img, 1280, 720, image::imageops::FilterType::Triangle)
        }
        Err(_) => scene(1280, 720),
    };
    let boxes: Vec<BBox> = (0..12)
        .map(|i| {
            let (x, y) = ((i * 97 % 1160) as f32, (i * 53 % 630) as f32);
            BBox { class: 0, x1: x, y1: y, x2: x + 110.0, y2: y + 80.0 }
        })
        .collect();
    for c in Condition::ALL {
        bench(c.id(), 15, || conditions::apply(c, 3, &img, &boxes, 42).unwrap());
    }
    let jpeg = conditions::encode_jpeg(&img).unwrap();
    bench("encode_jpeg (q95)", 15, || conditions::encode_jpeg(&img).unwrap());
    bench("decode (jpeg)", 15, || conditions::decode(&jpeg).unwrap());
    bench("render haze 3 (end to end)", 15, || {
        conditions::render(&jpeg, Condition::Haze, 3, &boxes, 42).unwrap()
    });
    for s in [1, 5] {
        for c in [Condition::Defocus, Condition::MotionBlur] {
            bench(&format!("{} {s}", c.id()), 9, || conditions::apply(c, s, &img, &boxes, 42).unwrap());
        }
    }
}
