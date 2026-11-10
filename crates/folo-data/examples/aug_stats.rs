//! Statistics of the training transform's output (docs/MODEL.md §8.6), to compare augmentation with another
//! trainer's: boxes per image and the distribution of box sizes after mosaic + affine (or letterbox + affine with
//! `--no-mosaic`), in output pixels.
//!
//!     cargo run --release -p folo-data --example aug_stats -- <dataset> [imgsz] [samples] [--no-mosaic]
//!
//! Prints one JSON object: `boxes_per_image` (mean, and the share of empty images), and quantiles of
//! `sqrt(w · h)`, `w` and `h` over all boxes, plus the share of boxes with `sqrt(w · h)` below 8, 16 and 32 px.

use folo_data::{AugmentConfig, Dataset, Rng, TrainTransform, open};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = args.first().expect("usage: aug_stats <dataset> [imgsz] [samples] [--no-mosaic]");
    let num =
        |i: usize, d: u32| args.get(i).filter(|a| !a.starts_with("--")).map_or(d, |a| a.parse().unwrap());
    let (imgsz, samples) = (num(1, 320), num(2, 4000) as usize);
    let mosaic = !args.iter().any(|a| a == "--no-mosaic");
    let ds = open(path).expect("open dataset");
    let train = &ds.train;
    let cfg = AugmentConfig { mosaic: if mosaic { 1.0 } else { 0.0 }, ..AugmentConfig::default() };
    let t = TrainTransform::train(imgsz, cfg, 0, 0);
    let (mut counts, mut side, mut ws, mut hs) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for k in 0..samples {
        let mut rng = Rng::new(0x5eed ^ k as u64);
        let i = rng.below(train.len());
        let out = t.apply_dataset(train, i, &mut rng).expect("transform");
        counts.push(out.sample.boxes.len() as f64);
        for b in &out.sample.boxes {
            let (w, h) = (b.w * imgsz as f32, b.h * imgsz as f32);
            side.push((w * h).sqrt() as f64);
            ws.push(w as f64);
            hs.push(h as f64);
        }
    }
    let q = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.total_cmp(b));
        let at = |p: f64| v[((v.len() - 1) as f64 * p).round() as usize];
        format!("[{:.2}, {:.2}, {:.2}, {:.2}, {:.2}]", at(0.05), at(0.25), at(0.5), at(0.75), at(0.95))
    };
    let share = |v: &[f64], lim: f64| v.iter().filter(|&&x| x < lim).count() as f64 / v.len() as f64;
    let mean = counts.iter().sum::<f64>() / counts.len() as f64;
    let empty = share(&counts, 0.5);
    let small = [share(&side, 8.0), share(&side, 16.0), share(&side, 32.0)];
    println!(
        "{{\"samples\": {samples}, \"mosaic\": {mosaic}, \"boxes_per_image\": {mean:.3}, \"empty\": {empty:.4}, \
         \"boxes\": {}, \"side_q05_25_50_75_95\": {}, \"w_q\": {}, \"h_q\": {}, \"side_lt_8_16_32\": [{:.4}, {:.4}, \
         {:.4}]}}",
        side.len(),
        q(&mut side),
        q(&mut ws),
        q(&mut hs),
        small[0],
        small[1],
        small[2]
    );
}
