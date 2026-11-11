//! Model quality on demand: a model's evaluation, predictions for an image, and the plausible
//! detections both are made of.

use crate::api::*;
use crate::synth::{Rng, hash};

use super::MockBackend;

impl MockBackend {
    /// [`Backend::model_eval`]'s answer.
    pub(super) fn eval_for(&self, id: &str) -> ApiResult<ModelEval> {
        self.check()?;
        let m = self
            .models
            .iter()
            .find(|m| m.id == id)
            .ok_or_else(|| ApiError::new("model_not_found", format!("No model {id}")))?;
        let metrics = m.metrics.unwrap_or(ModelMetrics {
            map50: 0.5,
            map50_95: 0.3,
            precision: 0.5,
            recall: 0.5,
            best_f1_conf: None,
        });
        // What is evaluated: the model's run's dataset (else one with its classes), its val split (train when
        // it has none, saying so).
        let run_ds = {
            let runs = self.runs.lock().unwrap();
            runs.iter()
                .find(|f| m.run_id.as_deref() == Some(&*f.run.id))
                .map(|f| f.run.config.dataset_id.clone())
        };
        let ds = run_ds
            .and_then(|id| self.fixture(&id).ok())
            .or_else(|| self.datasets.iter().find(|d| d.ds.classes == m.classes))
            .map(|d| &d.ds);
        let split = if ds.is_some_and(|d| d.splits.val == 0) { Split::Train } else { Split::Val };
        let images =
            ds.map_or(800, |d| if split == Split::Val { d.splits.val } else { d.splits.train }) as u32;
        let split_fallback = (split != Split::Val).then(|| {
            folo_api::SplitFallback::empty_split(
                Split::Val,
                split,
                ds.map_or("?", |d| &d.name),
                images as u64,
            )
        });
        let n = m.classes.len();
        let mut rng = Rng::new(hash(id));
        // Each class's PR curve has its AP50 as its area: precision 1 - (r / r1)^k up to the
        // recall r1 it reaches (k = 6, flat then falling, as detector curves run; a class too
        // good for that reaches recall 1 with a steeper k). Its precision and recall are the
        // curve's best-F1 point, so the dots sit on their curves.
        let curve = |ap: f32| {
            let (r1, k) =
                if ap * 7. / 6. <= 1. { (ap * 7. / 6., 6.) } else { (1., ap / (1. - ap).max(1e-3)) };
            move |r: f32| if r > r1 { 0. } else { 1. - (r / r1).powf(k) }
        };
        // Per-class spread around the model's figures, scaled so the classes average to them
        // (the legend's "All classes" row is their mean).
        let spread: Vec<f32> = (0..n).map(|c| 1.0 + 0.07 * rng.normal() - 0.025 * c as f32).collect();
        let mean = (spread.iter().sum::<f32>() / n.max(1) as f32).max(1e-3);
        let per_class: Vec<ClassMetrics> = (0..n as u32)
            .map(|c| {
                let d = spread[c as usize] / mean;
                let ap = (metrics.map50 * d).min(0.995);
                let p_at = curve(ap);
                let (recall, precision) = (1..=200)
                    .map(|i| {
                        let r = i as f32 / 200.;
                        (r, p_at(r))
                    })
                    .max_by(|a, b| {
                        let f1 = |(r, p): &(f32, f32)| 2. * r * p / (r + p).max(1e-6);
                        f1(a).total_cmp(&f1(b))
                    })
                    .unwrap_or((metrics.recall, metrics.precision));
                ClassMetrics {
                    class: c,
                    map50: ap,
                    map50_95: (metrics.map50_95 * d).min(0.99),
                    precision,
                    recall,
                }
            })
            .collect();
        let pr = per_class
            .iter()
            .map(|cm| {
                let p_at = curve(cm.map50);
                // Sampled up to where the curve ends, its end point (precision 0 at r1) included.
                let r1 = if cm.map50 * 7. / 6. <= 1. { cm.map50 * 7. / 6. } else { 1. };
                let mut recall: Vec<f32> =
                    (0..=100).map(|i| i as f32 / 100.).take_while(|&r| r == 0. || r < r1).collect();
                recall.push(r1);
                let precision: Vec<f32> =
                    recall.iter().map(|&r| if r >= r1 { 0. } else { p_at(r) }).collect();
                PrCurve { class: cm.class, recall, precision, ap50: cm.map50 }
            })
            .collect();
        let mut confusion = vec![vec![0u32; n + 1]; n + 1];
        for t in 0..n {
            let count = 900 + rng.below(500);
            let recall = per_class[t].recall;
            let hit = (count as f32 * recall) as u32;
            confusion[t][t] = hit;
            let mut left = count - hit;
            for (p, row_cell) in confusion[t].iter_mut().enumerate().take(n) {
                if p != t {
                    let share = (left as f32 * rng.range(0.05, 0.25)) as u32;
                    *row_cell += share;
                    left -= share;
                }
            }
            confusion[t][n] = left;
            confusion[n][t] = (count as f32 * (1.0 - per_class[t].precision) * 0.6) as u32;
        }
        Ok(ModelEval {
            pr,
            confusion,
            per_class,
            images,
            best_f1_conf: metrics.best_f1_conf,
            dataset_id: ds.map(|d| d.id.clone()),
            dataset_name: ds.map(|d| d.name.clone()),
            split: Some(split),
            split_fallback,
            // folo-vision's confusion thresholds (docs/METRICS.md §7).
            confusion_conf: Some(0.25),
            confusion_iou: Some(0.45),
        })
    }

    /// [`Backend::predict`]'s answer.
    pub(super) fn predict_for(&self, req: &PredictRequest) -> ApiResult<PredictResponse> {
        self.check()?;
        let m = self
            .models
            .iter()
            .find(|m| m.id == req.model_id)
            .ok_or_else(|| ApiError::new("model_not_found", format!("No model {}", req.model_id)))?;
        let q = m.metrics.as_ref().map_or(0.5, |m| m.map50_95 / 0.8).clamp(0.1, 1.0);
        let (w, h, raw, seed) = match &req.input {
            PredictInput::Dataset { dataset_id, image_id } => {
                let f = self.fixture(dataset_id)?;
                let im = f
                    .images
                    .iter()
                    .find(|im| &im.id == image_id)
                    .ok_or_else(|| ApiError::new("image_not_found", format!("No image {image_id}")))?;
                let p = fake_predictions(
                    &im.boxes,
                    im.width,
                    im.height,
                    q,
                    &m.classes,
                    hash(image_id) ^ hash(&m.id),
                );
                (im.width, im.height, p, hash(image_id))
            }
            PredictInput::File { name, bytes } => {
                let (w, h) = image::ImageReader::new(std::io::Cursor::new(bytes.as_slice()))
                    .with_guessed_format()
                    .ok()
                    .and_then(|r| r.into_dimensions().ok())
                    .ok_or_else(|| {
                        ApiError::new("unsupported_image", format!("Could not decode {name}"))
                            .with_hint("FOLO reads JPEG and PNG images.")
                    })?;
                let seed = bytes.iter().step_by(97).fold(hash(name), |h, &b| h.rotate_left(5) ^ b as u64);
                let mut rng = Rng::new(seed);
                let n = 2 + rng.below(4);
                let s = w.min(h) as f32;
                let p = (0..n)
                    .map(|_| {
                        let bw = rng.range(0.12, 0.32) * s;
                        let bh = bw * rng.range(0.7, 1.4);
                        let x1 = rng.range(0.0, w as f32 - bw);
                        let y1 = rng.range(0.0, (h as f32 - bh).max(1.0));
                        let class = rng.below(m.classes.len() as u32);
                        ScoredBox {
                            class,
                            name: m.classes.get(class as usize).cloned().unwrap_or_default(),
                            score: rng.range(0.3, 0.92),
                            x1,
                            y1,
                            x2: x1 + bw,
                            y2: (y1 + bh).min(h as f32),
                        }
                    })
                    .collect();
                (w, h, p, seed)
            }
        };
        let mut boxes: Vec<ScoredBox> = raw.into_iter().filter(|b| b.score >= req.conf).collect();
        boxes.sort_by(|a, b| b.score.total_cmp(&a.score));
        boxes.truncate(req.max_det as usize);
        let scale = match m.arch.as_str() {
            "folo-n" => 1.0,
            "folo-s" => 1.7,
            "folo-m" => 3.1,
            _ => 5.4,
        } * (m.imgsz as f32 / 640.0).powi(2);
        let mut rng = Rng::new(seed ^ 0x71);
        let pre = 0.62 + 0.1 * rng.f32();
        let inf = 2.9 * scale * (1.0 + 0.05 * rng.normal());
        let post = 0.14 + 0.02 * boxes.len() as f32;
        Ok(PredictResponse {
            model_id: m.id.clone(),
            image: ImageSize { width: w, height: h },
            boxes,
            timing: Timing {
                preprocess_ms: pre as f64,
                inference_ms: inf as f64,
                postprocess_ms: post as f64,
                total_ms: (pre + inf + post) as f64,
            },
            device: "gpu:0".into(),
            cold: false,
        })
    }
}

pub(super) fn fake_predictions(
    truth: &[BBoxT],
    w: u32,
    h: u32,
    q: f32,
    names: &[String],
    seed: u64,
) -> Vec<ScoredBox> {
    let classes = names.len().max(1) as u32;
    let name = |c: u32| names.get(c as usize).cloned().unwrap_or_default();
    let mut rng = Rng::new(seed);
    let mut out = Vec::new();
    for b in truth {
        // A model can't find a class it wasn't trained on (a 3-class model on a 5-class image).
        if b.class >= classes || rng.f32() > 0.55 + 0.44 * q {
            continue;
        }
        let (bw, bh) = (b.x2 - b.x1, b.y2 - b.y1);
        let j = (1.0 - q) * 0.12 + 0.015;
        let x1 = (b.x1 + rng.normal() * j * bw).clamp(0.0, w as f32);
        let y1 = (b.y1 + rng.normal() * j * bh).clamp(0.0, h as f32);
        let x2 = (b.x2 + rng.normal() * j * bw).clamp(x1 + 1.0, w as f32);
        let y2 = (b.y2 + rng.normal() * j * bh).clamp(y1 + 1.0, h as f32);
        let class = if rng.f32() < (1.0 - q) * 0.18 { rng.below(classes) } else { b.class };
        out.push(ScoredBox {
            class,
            name: name(class),
            score: (0.45 + 0.5 * q * rng.range(0.75, 1.0) + 0.04 * rng.normal()).clamp(0.26, 0.99),
            x1,
            y1,
            x2,
            y2,
        });
    }
    let fps = if rng.f32() < (1.0 - q) * 0.9 { 1 + rng.below(2) } else { 0 };
    for _ in 0..fps {
        let s = w.min(h) as f32 * rng.range(0.05, 0.15);
        let x1 = rng.range(0.0, w as f32 - s);
        let y1 = rng.range(0.0, h as f32 - s);
        let class = rng.below(classes);
        out.push(ScoredBox {
            class,
            name: name(class),
            score: rng.range(0.26, 0.48),
            x1,
            y1,
            x2: x1 + s,
            y2: y1 + s,
        });
    }
    out
}

pub(super) type BBoxT = super::types::BBox;
