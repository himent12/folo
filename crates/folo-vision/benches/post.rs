//! Post-processing and evaluation benchmarks. Run with `cargo bench -p folo-vision --bench post`.
//!
//! - decode + NMS on one 640x640 image's head output (8400 anchors x 80 classes), predict and val settings;
//! - the NMS-free top-k of the one-to-one head (docs/MODEL.md §6.6) on the same decoded outputs;
//! - batched NMS alone on 8400 boxes;
//! - the task-aligned assigner on 8400 anchors x 20 gts;
//! - the evaluator on 5,000 images of random data.

#[path = "../tests/common/mod.rs"]
mod common;

use common::Rng;
use folo_vision::assign::{TalConfig, assign};
use folo_vision::cocoeval::{CocoEval, CocoTruth};
use folo_vision::decode::{decode_head, make_anchors};
use folo_vision::nms::{batched_nms, non_max_suppression, topk_detections};
use folo_vision::{Detection, EvalConfig, Evaluator, NmsConfig, Truth, Xyxy};
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
    let med = times[times.len() / 2];
    let min = times[0];
    println!("{name:<52} median {:>9.3} ms   min {:>9.3} ms   ({iters} iters)", ms(med), ms(min));
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn main() {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    println!("folo-vision post bench, {threads} threads available");
    let mut rng = Rng::new(0xBE7C);
    let (nc, reg_max) = (80usize, 16usize);
    let anchors = make_anchors(640, 640, &[8, 16, 32], 0.5);
    let a = anchors.len();

    // Raw head outputs: DFL logits peaked around random distances; class logits mostly very negative with a few
    // confident anchors, like a trained model's output.
    let mut reg = vec![0.0f32; a * 4 * reg_max];
    for row in reg.chunks_exact_mut(reg_max) {
        let peak = rng.below(reg_max);
        for (i, v) in row.iter_mut().enumerate() {
            *v = -((i as f32 - peak as f32).powi(2)) * 0.5 + rng.range(-0.2, 0.2);
        }
    }
    let cls: Vec<f32> = (0..a * nc)
        .map(|_| if rng.below(200) == 0 { rng.range(-2.0, 4.0) } else { rng.range(-12.0, -4.0) })
        .collect();

    bench("decode_head (8400 x (64 + 80))", 50, || decode_head(&reg, &cls, &anchors, reg_max, nc));
    let (boxes, scores) = decode_head(&reg, &cls, &anchors, reg_max, nc);

    let predict = NmsConfig::default();
    let n = non_max_suppression(&boxes, &scores, nc, &predict).len();
    bench(&format!("nms predict (conf 0.25, best class) -> {n}"), 50, || {
        non_max_suppression(&boxes, &scores, nc, &predict)
    });
    let val = NmsConfig::val();
    let n = non_max_suppression(&boxes, &scores, nc, &val).len();
    bench(&format!("nms val (conf 0.001, multi-label) -> {n}"), 50, || {
        non_max_suppression(&boxes, &scores, nc, &val)
    });

    // Worst case: uniform random scores, every (anchor, class) pair above conf, capped at 30k candidates.
    let uniform: Vec<f32> = (0..a * nc).map(|_| rng.f32()).collect();
    let n = non_max_suppression(&boxes, &uniform, nc, &val).len();
    bench(&format!("nms val, uniform scores (672k pairs -> 30k) -> {n}"), 20, || {
        non_max_suppression(&boxes, &uniform, nc, &val)
    });

    // The one-to-one head's selection (no NMS) on the same outputs.
    let n = topk_detections(&boxes, &scores, nc, &predict).len();
    bench(&format!("o2o top-k predict (conf 0.25, best class) -> {n}"), 50, || {
        topk_detections(&boxes, &scores, nc, &predict)
    });
    let n = topk_detections(&boxes, &scores, nc, &val).len();
    bench(&format!("o2o top-k val (conf 0.001, pairs) -> {n}"), 50, || {
        topk_detections(&boxes, &scores, nc, &val)
    });
    let n = topk_detections(&boxes, &uniform, nc, &val).len();
    bench(&format!("o2o top-k val, uniform scores -> {n}"), 20, || {
        topk_detections(&boxes, &uniform, nc, &val)
    });

    // Batched NMS alone on 8400 random boxes / classes.
    let rb: Vec<Xyxy> = (0..a).map(|_| rng.bbox(640.0, 8.0, 160.0)).collect();
    let rs: Vec<f32> = (0..a).map(|_| rng.f32()).collect();
    let rc: Vec<u32> = (0..a).map(|_| rng.below(nc) as u32).collect();
    bench("batched_nms 8400 boxes, 80 classes, max_det 300", 50, || {
        batched_nms(&rb, &rs, &rc, 0.7, 300, false)
    });
    bench("batched_nms 8400 boxes, agnostic, max_det 300", 50, || batched_nms(&rb, &rs, &rc, 0.7, 300, true));
    bench("batched_nms 8400 boxes, 80 classes, no max_det", 50, || {
        batched_nms(&rb, &rs, &rc, 0.7, usize::MAX, false)
    });

    // Assigner: 8400 anchors, 20 gts.
    let gts: Vec<Xyxy> = (0..20).map(|_| rng.bbox(640.0, 16.0, 320.0)).collect();
    let labels: Vec<u32> = (0..20).map(|_| rng.below(nc) as u32).collect();
    let pts = anchors.points_px();
    let probs: Vec<f32> = cls.iter().map(|&x| folo_vision::decode::sigmoid(x)).collect();
    bench("TAL assign 8400 anchors x 20 gts", 50, || {
        assign(&probs, nc, &boxes, &pts, &gts, &labels, &TalConfig::default())
    });

    // Evaluator: 5,000 images, ~7 gts each, 20 jittered TPs-ish + random FPs, 80 classes.
    let images: Vec<(Vec<Detection>, Vec<Truth>)> = (0..5000)
        .map(|_| {
            let truths: Vec<Truth> = (0..1 + rng.below(12))
                .map(|_| Truth { bbox: rng.bbox(640.0, 8.0, 300.0), class: rng.below(nc) as u32 })
                .collect();
            let mut preds: Vec<Detection> = Vec::new();
            for t in &truths {
                for _ in 0..3 {
                    preds.push(Detection {
                        bbox: rng.jitter(&t.bbox, 0.1),
                        score: rng.f32(),
                        class: t.class,
                    });
                }
            }
            for _ in 0..rng.below(80) {
                preds.push(Detection {
                    bbox: rng.bbox(640.0, 8.0, 300.0),
                    score: rng.f32() * 0.5,
                    class: rng.below(nc) as u32,
                });
            }
            (preds, truths)
        })
        .collect();
    let total_preds: usize = images.iter().map(|i| i.0.len()).sum();
    let total_truths: usize = images.iter().map(|i| i.1.len()).sum();
    println!("evaluator data: 5000 images, {total_preds} preds, {total_truths} truths");
    bench("evaluator add_image x 5000", 10, || {
        let mut ev = Evaluator::new(nc);
        for (p, t) in &images {
            ev.add_image(p, t);
        }
        ev
    });
    bench("evaluator add_image x 5000 (coco_compatible)", 10, || {
        let mut ev = Evaluator::with_config(nc, EvalConfig::coco());
        for (p, t) in &images {
            ev.add_image(p, t);
        }
        ev
    });
    let mut ev = Evaluator::new(nc);
    for (p, t) in &images {
        ev.add_image(p, t);
    }
    bench("evaluator compute (5000 images)", 10, || ev.compute());
    let r = ev.compute();
    println!("  map50 {:.4}  map50_95 {:.4}  P {:.4}  R {:.4}", r.map50, r.map50_95, r.precision, r.recall);

    let coco_truths: Vec<Vec<CocoTruth>> = images
        .iter()
        .map(|(_, ts)| ts.iter().map(|t| CocoTruth::from_box(t.bbox, t.class)).collect())
        .collect();
    bench("cocoeval add_image x 5000", 10, || {
        let mut cev = CocoEval::new(nc);
        for (i, ((p, _), t)) in images.iter().zip(&coco_truths).enumerate() {
            cev.add_image(i as u64 + 1, p, t);
        }
        cev
    });
    let mut cev = CocoEval::new(nc);
    for (i, ((p, _), t)) in images.iter().zip(&coco_truths).enumerate() {
        cev.add_image(i as u64 + 1, p, t);
    }
    bench("cocoeval compute (5000 images)", 10, || cev.compute());
    let cs = cev.compute();
    println!("  coco ap {:.4}  ap50 {:.4}  ar100 {:.4}", cs.ap, cs.ap50, cs.ar100);
}
