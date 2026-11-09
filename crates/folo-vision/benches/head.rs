//! End-to-end prediction with the one-to-one head (top-k, no NMS) against the one-to-many head with NMS on the same
//! trained `head = "o2o"` model (docs/MODEL.md §6.6), the numbers in docs/BENCHMARKS.md's NMS-free section. Run with
//! `FOLO_BENCH_WEIGHTS=<o2o .folo> FOLO_BENCH_IMAGES=<folder of images> cargo bench -p folo-vision --bench head`
//! (`FOLO_BENCH_DEVICE=cpu,gpu` the devices, default both where available; `FOLO_BENCH_IMGSZ` the letterbox size,
//! default 640; `RAYON_NUM_THREADS` the CPU pool).
//!
//! Trained weights matter here: an untrained head scores nothing above `conf`, so NMS would look free. The
//! predictor runs `predict` (batch 1, `rect`) and `predict_batch` (batch 16, square) on decoded `u8` images at
//! the predict defaults (`conf 0.25`, `iou 0.7`, `max_det 300`); the two heads alternate run by run, so a load
//! change on the shared machine hits both. Reported: the median wall time per call and the median of the
//! predictor's own `postprocess_ms` (decode, readback, NMS or top-k, rescale), per image.

use std::path::PathBuf;
use std::time::Instant;

use folo_core::Device;
use folo_data::Image;
use folo_vision::model::{FoloModel, HeadMode};
use folo_vision::predict::{PredictConfig, Predictor};

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn images(dir: &PathBuf, n: usize) -> Vec<Image> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .expect("FOLO_BENCH_IMAGES is a folder")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| matches!(x.to_str(), Some("jpg" | "jpeg" | "png"))))
        .collect();
    files.sort();
    files.iter().take(n).map(|p| folo_data::read_image(p).expect("image")).collect()
}

fn main() {
    let (Some(weights), Some(dir)) =
        (std::env::var_os("FOLO_BENCH_WEIGHTS"), std::env::var_os("FOLO_BENCH_IMAGES"))
    else {
        println!(
            "set FOLO_BENCH_WEIGHTS (an o2o .folo) and FOLO_BENCH_IMAGES (a folder of images); skipping"
        );
        return;
    };
    let imgsz: u32 = std::env::var("FOLO_BENCH_IMGSZ").ok().and_then(|s| s.parse().ok()).unwrap_or(640);
    let devices = std::env::var("FOLO_BENCH_DEVICE").unwrap_or_else(|_| "cpu,gpu".into());
    let imgs = images(&dir.into(), 16);
    assert_eq!(imgs.len(), 16, "FOLO_BENCH_IMAGES needs at least 16 images");
    for dev in devices.split(',') {
        let device = match dev.trim() {
            "cpu" => Device::Cpu,
            "gpu" => match folo_gpu::init() {
                Ok(d) => d,
                Err(e) => {
                    println!("gpu: {e}; skipping");
                    continue;
                }
            },
            other => panic!("unknown device {other}"),
        };
        let model = FoloModel::open_on(&weights, device).expect("weights");
        assert!(model.has_o2o(), "FOLO_BENCH_WEIGHTS must be an o2o model (folo train --head o2o)");
        let threads = rayon::current_num_threads();
        println!("{device} ({threads} rayon threads), imgsz {imgsz}, {} classes", model.num_classes());
        let cfg = PredictConfig { imgsz, ..PredictConfig::default() };
        let mut pred = Predictor::new(model, cfg.clone()).expect("predictor");
        for (batch, iters) in [(1usize, 60usize), (16, 12)] {
            let mut wall = [Vec::new(), Vec::new()];
            let mut post = [Vec::new(), Vec::new()];
            let mut boxes = [0usize; 2];
            for it in 0..iters + 3 {
                for (k, head) in [HeadMode::O2o, HeadMode::O2m].into_iter().enumerate() {
                    pred.set_config(PredictConfig { head: Some(head), ..cfg.clone() }).expect("config");
                    let t = Instant::now();
                    let out = if batch == 1 {
                        vec![pred.predict(&imgs[it % 16]).expect("predict")]
                    } else {
                        pred.predict_batch(&imgs).expect("predict")
                    };
                    let ms = t.elapsed().as_secs_f64() * 1e3;
                    if it >= 3 {
                        wall[k].push(ms);
                        post[k].push(out.iter().map(|p| p.timing.postprocess_ms).sum::<f64>() / batch as f64);
                        boxes[k] += out.iter().map(|p| p.detections.len()).sum::<usize>();
                    }
                }
            }
            for (k, name) in ["o2o (top-k, no NMS)", "o2m + NMS"].into_iter().enumerate() {
                let n = wall[k].len();
                let w = median(&mut wall[k]);
                let p = median(&mut post[k]);
                let per_img = boxes[k] as f64 / (n * batch) as f64;
                println!(
                    "  b{batch:<2} {name:<20} median {w:>8.2} ms/call ({:>6.1} img/s)  postprocess {p:>6.3} ms/img  \
                     {per_img:>5.1} boxes/img",
                    1e3 * batch as f64 / w
                );
            }
        }
    }
}
