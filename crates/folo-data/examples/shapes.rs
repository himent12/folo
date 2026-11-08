//! Writes the synthetic shapes dataset of docs/MODEL.md §10.6 (the convergence smoke test) as a YOLO folder:
//! 4 classes (circle, square, triangle, star), 2000 train + 200 val square images at 320 px, 1 to 8 objects per
//! image, seed 0.
//!
//! ```sh
//! cargo run --release -p folo-data --example shapes -- D:/data/shapes-smoke
//! cargo run --release -p folo-data --example shapes -- <dir> [train=2000] [val=200] [imgsz=320] [seed=0]
//! ```

use folo_data::ImageFormat;
use folo_data::synthetic::{SyntheticConfig, write_yolo};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(dir) = args.first() else {
        eprintln!("usage: shapes <dir> [train=2000] [val=200] [imgsz=320] [seed=0]");
        std::process::exit(2);
    };
    let num = |i: usize, default: u64| args.get(i).map_or(default, |s| s.parse().expect("a number"));
    let (train, val, imgsz, seed) = (num(1, 2000), num(2, 200), num(3, 320), num(4, 0));
    let images = (train + val) as usize;
    let cfg = SyntheticConfig {
        images,
        classes: ["circle", "square", "triangle", "star"].map(String::from).to_vec(),
        imgsz: imgsz as u32,
        seed,
        min_objects: 1,
        max_objects: 8,
        empty_fraction: 0.0,
        vary_aspect: false,
        val_fraction: val as f32 / images as f32,
        ..SyntheticConfig::default()
    };
    assert_eq!(cfg.train_count() as u64, train, "split rounding");
    let t = std::time::Instant::now();
    let w = write_yolo(&cfg, std::path::Path::new(dir), ImageFormat::Jpeg(95)).expect("write dataset");
    println!(
        "wrote {} train + {} val images ({} boxes) to {} in {:.1}s",
        w.splits[0],
        w.splits[1],
        w.boxes,
        w.yaml.display(),
        t.elapsed().as_secs_f64()
    );
}
