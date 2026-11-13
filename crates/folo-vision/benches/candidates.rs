//! Candidate-first prediction (docs/MODEL.md §9.5) on real images with trained weights. Run with
//! `cargo bench -p folo-vision --bench candidates`.
//!
//! `FOLO_BENCH_WEIGHTS` (a `.folo` file) and `FOLO_BENCH_IMAGES` (an image folder, e.g. COCO val2017) are required;
//! `FOLO_BENCH_IMAGES_N` (default: all) caps the images, by name. Each image is letterboxed as the CPU predictor
//! does (rect, 640) and predicted at `FOLO_BENCH_CONF` (default 0.25) with the model's head and the box fusion
//! `FOLO_BENCH_FUSION` (`none`, the default, `vote` or `dfl`; MODEL.md §9.4):
//!
//! - **Statistics**: the candidates that need a box, per level and in total, per image: median, p90, p99, max,
//!   mean. With `FOLO_BENCH_VERIFY=1` every image's detections are also compared with the dense path's (forward,
//!   then `HeadOutput::detections`), bit for bit.
//! - **Sweep** (`FOLO_BENCH_SWEEP=1`): on the first `FOLO_BENCH_SWEEP_N` (default 32) images, at confidence
//!   thresholds from 0.5 down to 0.001, the model step candidate-first (`predict_candidates` + `finish_candidates`)
//!   against dense (`forward` + `detections`), alternated image by image for `FOLO_BENCH_ROUNDS` (default 3)
//!   rounds: the mean candidates per level and the median time per image of both, which locates the break-even.
//!
//! `FOLO_BENCH_DEVICE=gpu` runs all of it on the GPU (both paths end in a readback, so the times are synchronised)
//! in `FOLO_BENCH_PRECISION` (`f32` or `f16`; tensor cores as `FOLO_GPU_TENSOR_CORES` says).

use std::hint::black_box;
use std::time::Instant;

use folo_core::Device;
use folo_core::{Tensor, no_grad};
use folo_vision::fusion::BoxFusion;
use folo_vision::nms::{Detection, NmsConfig};
use folo_vision::predict::{Precision, PredictConfig, Predictor};

fn env<T: std::str::FromStr>(k: &str) -> Option<T> {
    std::env::var(k).ok().and_then(|s| s.parse().ok())
}

fn quantile(v: &mut [usize], q: f64) -> usize {
    v.sort_unstable();
    v[((v.len() as f64 * q) as usize).min(v.len() - 1)]
}

fn bits(d: &[Detection]) -> Vec<[u32; 6]> {
    d.iter()
        .map(|d| {
            let b = d.bbox;
            [d.class, d.score.to_bits(), b.x1.to_bits(), b.y1.to_bits(), b.x2.to_bits(), b.y2.to_bits()]
        })
        .collect()
}

fn main() {
    let (Some(weights), Some(dir)) =
        (env::<String>("FOLO_BENCH_WEIGHTS"), env::<String>("FOLO_BENCH_IMAGES"))
    else {
        println!(
            "candidates bench skipped: set FOLO_BENCH_WEIGHTS (a .folo file) and FOLO_BENCH_IMAGES (an image folder)"
        );
        return;
    };
    let conf: f32 = env("FOLO_BENCH_CONF").unwrap_or(0.25);
    let mut paths: Vec<std::path::PathBuf> = std::fs::read_dir(&dir)
        .expect("FOLO_BENCH_IMAGES")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension().is_some_and(|e| e.eq_ignore_ascii_case("jpg") || e.eq_ignore_ascii_case("png"))
        })
        .collect();
    paths.sort();
    paths.truncate(env("FOLO_BENCH_IMAGES_N").unwrap_or(usize::MAX));
    let gpu = env::<String>("FOLO_BENCH_DEVICE").is_some_and(|d| d == "gpu");
    let prec: Precision = env("FOLO_BENCH_PRECISION").unwrap_or(Precision::F32);
    let dev =
        if gpu { folo_gpu::init().expect("FOLO_BENCH_DEVICE=gpu: no GPU adapter") } else { Device::Cpu };
    let pred = Predictor::open_on(&weights, PredictConfig::default(), dev, prec).expect("weights");
    let model = pred.model();
    let fusion = match env::<String>("FOLO_BENCH_FUSION").as_deref() {
        None | Some("none") => BoxFusion::None,
        Some("vote") => BoxFusion::Vote,
        Some("dfl") => BoxFusion::Dfl,
        Some(f) => panic!("FOLO_BENCH_FUSION: {f} is not none, vote or dfl"),
    };
    let nms = NmsConfig { conf, fusion, ..NmsConfig::default() };
    let verify = env::<String>("FOLO_BENCH_VERIFY").is_some_and(|v| v == "1");
    println!(
        "candidates bench: {weights} ({}, head {}) on {dev:?} {prec:?}{}, {} images from {dir}, conf {conf}, fusion {}, {} threads",
        model.arch(),
        model.head_mode(),
        if gpu {
            format!(" ({})", folo_gpu::GpuContext::shared().expect("context").tensor_cores_status())
        } else {
            String::new()
        },
        paths.len(),
        fusion.as_str(),
        rayon::current_num_threads()
    );
    let prepare = |p: &std::path::Path| -> Tensor {
        let img = folo_data::read_image(p).expect("image");
        pred.prepare(std::slice::from_ref(&img), true).expect("prepare").input
    };

    let _g = no_grad();
    let mut per_level: [Vec<usize>; 3] = Default::default();
    let mut total = Vec::new();
    let mut mismatched = 0;
    for p in &paths {
        let x = prepare(p);
        let (h, w) = (x.dims()[2], x.dims()[3]);
        let anchors = model.anchors(h, w);
        let cands = model.predict_candidates(&x, &nms).expect("candidates");
        let c = &cands[0];
        for (l, lv) in anchors.levels.iter().enumerate() {
            let range = lv.start as u32..(lv.start + lv.height * lv.width) as u32;
            per_level[l].push(c.cands.iter().filter(|c| range.contains(&c.anchor)).count());
        }
        total.push(c.cands.len());
        if verify {
            let lazy = model.finish_candidates(c, &nms);
            let dense = model.forward(&x).expect("forward").detections(&nms).expect("detections");
            if bits(&lazy) != bits(&dense[0]) {
                mismatched += 1;
                println!("MISMATCH {}", p.display());
            }
        }
    }
    for (name, v) in ["P3", "P4", "P5", "total"].iter().zip(per_level.iter_mut().chain([&mut total])) {
        let mean = v.iter().sum::<usize>() as f64 / v.len() as f64;
        println!(
            "candidates {name:<5} median {:>5}   p90 {:>5}   p99 {:>5}   max {:>5}   mean {mean:>7.1}",
            quantile(v, 0.5),
            quantile(v, 0.9),
            quantile(v, 0.99),
            quantile(v, 1.0),
        );
    }
    if verify {
        println!("verified {} images against the dense path: {mismatched} mismatched", paths.len());
        assert_eq!(mismatched, 0, "candidate-first detections differ from the dense path's");
    }

    if env::<String>("FOLO_BENCH_SWEEP").is_some_and(|v| v == "1") {
        let n = env("FOLO_BENCH_SWEEP_N").unwrap_or(32).min(paths.len());
        let rounds = env("FOLO_BENCH_ROUNDS").unwrap_or(3);
        let inputs: Vec<Tensor> = paths[..n].iter().map(|p| prepare(p)).collect();
        println!("sweep: {n} images, {rounds} rounds, times are the median per image of the model step");
        for conf in [0.5f32, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001] {
            let nms = NmsConfig { conf, fusion, ..NmsConfig::default() };
            let (mut lazy_t, mut dense_t, mut counts) = (Vec::new(), Vec::new(), [0usize; 3]);
            for r in 0..rounds {
                for (i, x) in inputs.iter().enumerate() {
                    let run_lazy = || {
                        let s = Instant::now();
                        let c = model.predict_candidates(x, &nms).expect("candidates");
                        let d = model.finish_candidates(&c[0], &nms);
                        (s.elapsed().as_secs_f64() * 1e3, c, d)
                    };
                    let run_dense = || {
                        let s = Instant::now();
                        let d = model.forward(x).expect("forward").detections(&nms).expect("detections");
                        (s.elapsed().as_secs_f64() * 1e3, d)
                    };
                    let ((tl, c, dl), (td, dd)) = if (r + i) % 2 == 0 {
                        let l = run_lazy();
                        (l, run_dense())
                    } else {
                        let d = run_dense();
                        (run_lazy(), d)
                    };
                    assert_eq!(bits(&dl), bits(&dd[0]), "conf {conf}: candidate-first and dense differ");
                    if r == 0 {
                        let (h, w) = (x.dims()[2], x.dims()[3]);
                        for (l, lv) in model.anchors(h, w).levels.iter().enumerate() {
                            let range = lv.start as u32..(lv.start + lv.height * lv.width) as u32;
                            counts[l] += c[0].cands.iter().filter(|c| range.contains(&c.anchor)).count();
                        }
                    }
                    lazy_t.push(black_box(tl));
                    dense_t.push(td);
                }
            }
            let med = |v: &mut Vec<f64>| {
                v.sort_by(f64::total_cmp);
                v[v.len() / 2]
            };
            let (l, d) = (med(&mut lazy_t), med(&mut dense_t));
            let c = counts.map(|c| c as f64 / n as f64);
            println!(
                "conf {conf:<6} candidates P3 {:>7.1} P4 {:>6.1} P5 {:>6.1}   candidate-first {l:>6.2} ms   dense {d:>6.2} ms   ratio {:.3}",
                c[0],
                c[1],
                c[2],
                l / d
            );
        }
    }
}
