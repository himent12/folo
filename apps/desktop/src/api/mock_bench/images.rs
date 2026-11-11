//! A benchmark's images: where they come from (a dataset split or a labeling job), their pages, and
//! their pixels under a condition.

use std::sync::Arc;

use folo_bench::Condition;

use crate::api::mock::DatasetFixture;
use crate::api::*;
use crate::synth::{Rng, Scene, hash};

use super::{BenchImgSource, MockBench, find_bench};

impl MockBench {
    pub fn images(
        &self,
        id: &str,
        offset: u32,
        limit: u32,
        condition: Option<(&str, u8)>,
        datasets: &[DatasetFixture],
        models: &[Model],
    ) -> ApiResult<BenchImagePage> {
        let parsed = parse_cond_query(condition)?;
        self.with(datasets, models, |d, _, _| {
            let b = find_bench(&d.benches, id)?;
            let total = b.images.len() as u64;
            let lim = if limit == 0 { 100 } else { limit.min(1000) } as usize;
            let seed = b.bench.seed;
            let images = b
                .images
                .iter()
                .skip(offset as usize)
                .take(lim)
                .map(|im| {
                    let boxes = match parsed {
                        Some((Condition::Distance, sev)) => {
                            distance_boxes(&im.boxes, im.width, im.height, sev)
                        }
                        _ => im.boxes.clone(),
                    };
                    let _ = seed;
                    BenchImage {
                        id: im.id.clone(),
                        file: im.file.clone(),
                        width: im.width,
                        height: im.height,
                        boxes,
                    }
                })
                .collect();
            Ok(BenchImagePage { total, images })
        })
    }

    pub fn image_pixels(
        &self,
        id: &str,
        image_id: &str,
        condition: Option<(&str, u8)>,
        thumb: Option<u32>,
        datasets: &[DatasetFixture],
        models: &[Model],
    ) -> ApiResult<ImageData> {
        let parsed = parse_cond_query(condition)?;
        let (bench_seed, src) = self.with(datasets, models, |d, _, _| {
            let b = find_bench(&d.benches, id)?;
            let im = b
                .images
                .iter()
                .find(|i| i.id == image_id)
                .cloned()
                .ok_or_else(|| ApiError::new("image_not_found", format!("No image {image_id}")))?;
            Ok((b.bench.seed, im))
        })?;

        let (mut w, mut h) = (src.width, src.height);
        let mut scale = 1.0f32;
        if let Some(t) = thumb {
            let s = t as f32 / w.max(h) as f32;
            if s < 1.0 {
                scale = s;
                w = ((w as f32 * s).round() as u32).max(1);
                h = ((h as f32 * s).round() as u32).max(1);
            }
        }
        let cond_key = parsed.map_or((String::new(), 0), |(c, s)| (c.id().to_string(), s));
        let key = (id.to_string(), image_id.to_string(), cond_key.0, cond_key.1, w);
        if let Some((cw, ch, px)) = self.variant_cache.lock().unwrap().get(&key).cloned() {
            return Ok(ImageData::Rgba { width: cw, height: ch, pixels: px.to_vec() });
        }

        let rgba = src.scene().render(w, h);
        let pixels = match parsed {
            None => rgba,
            Some((cond, sev)) => {
                let mut rgb_raw = Vec::with_capacity((w * h * 3) as usize);
                for p in rgba.as_chunks::<4>().0 {
                    rgb_raw.extend_from_slice(&p[..3]);
                }
                let rgb = image::RgbImage::from_raw(w, h, rgb_raw)
                    .ok_or_else(|| ApiError::new("decode_failed", "Invalid scene dimensions"))?;
                let scaled_boxes: Vec<BBox> = src
                    .boxes
                    .iter()
                    .map(|b| BBox {
                        class: b.class,
                        x1: b.x1 * scale,
                        y1: b.y1 * scale,
                        x2: b.x2 * scale,
                        y2: b.y2 * scale,
                    })
                    .collect();
                let vseed = folo_bench::conditions::variant_seed(bench_seed, &src.id, cond, sev);
                let (out_rgb, _) = folo_bench::conditions::apply(cond, sev, &rgb, &scaled_boxes, vseed)
                    .map_err(|e| ApiError::new("invalid_query", e.to_string()))?;
                let mut out_rgba = Vec::with_capacity((w * h * 4) as usize);
                for p in out_rgb.as_raw().as_chunks::<3>().0 {
                    out_rgba.extend_from_slice(&[p[0], p[1], p[2], 255]);
                }
                out_rgba
            }
        };
        let mut cache = self.variant_cache.lock().unwrap();
        if cache.len() > 400 {
            cache.clear();
        }
        cache.insert(key, (w, h, Arc::new(pixels.clone())));
        Ok(ImageData::Rgba { width: w, height: h, pixels })
    }
}

pub(super) fn parse_cond_query(q: Option<(&str, u8)>) -> ApiResult<Option<(Condition, u8)>> {
    let Some((c, s)) = q else { return Ok(None) };
    if c.is_empty() {
        return Ok(None);
    }
    let cond = Condition::parse(c)
        .ok_or_else(|| ApiError::new("invalid_query", format!("Unknown condition `{c}`.")))?;
    if !(1..=BENCH_SEVERITIES).contains(&s) {
        return Err(ApiError::new("invalid_query", format!("Severity {s} must be 1..={BENCH_SEVERITIES}.")));
    }
    Ok(Some((cond, s)))
}

pub(super) fn distance_boxes(boxes: &[BBox], w: u32, h: u32, severity: u8) -> Vec<BBox> {
    const SCALES: [f32; 5] = [0.8, 0.65, 0.5, 0.375, 0.25];
    let k = SCALES[(severity.saturating_sub(1) as usize).min(4)];
    let (ox, oy) = (w as f32 * (1.0 - k) * 0.5, h as f32 * (1.0 - k) * 0.5);
    boxes
        .iter()
        .map(|b| BBox {
            class: b.class,
            x1: ox + b.x1 * k,
            y1: oy + b.y1 * k,
            x2: ox + b.x2 * k,
            y2: oy + b.y2 * k,
        })
        .collect()
}

pub(super) fn count_class_boxes(images: &[BenchImgSource], classes: usize) -> Vec<u64> {
    let mut counts = vec![0u64; classes];
    for im in images {
        for b in &im.boxes {
            if let Some(slot) = counts.get_mut(b.class as usize) {
                *slot += 1;
            }
        }
    }
    counts
}

pub(super) fn sample_source(pool: &[BenchImgSource], seed: u64, max_images: usize) -> Vec<BenchImgSource> {
    let mut order: Vec<usize> = (0..pool.len()).collect();
    let mut rng = Rng::new(seed ^ 0x5A17_BEEF);
    for i in (1..order.len()).rev() {
        let j = rng.below((i + 1) as u32) as usize;
        order.swap(i, j);
    }
    order
        .into_iter()
        .take(max_images)
        .enumerate()
        .map(|(idx, src_idx)| {
            let mut im = pool[src_idx].clone();
            im.id = (idx + 1).to_string();
            im
        })
        .collect()
}

pub(super) fn resolve_source(
    source: &BenchSource,
    datasets: &[DatasetFixture],
    jobs: &[LabelJob],
    default_split: Split,
) -> ApiResult<(BenchSource, String, Vec<String>, Vec<BenchImgSource>)> {
    match source {
        BenchSource::Dataset { dataset_id, split } => {
            let sp = split.unwrap_or(default_split);
            let f = datasets
                .iter()
                .find(|d| d.ds.id == *dataset_id)
                .ok_or_else(|| ApiError::new("dataset_not_found", format!("No dataset {dataset_id}")))?;
            let pool: Vec<BenchImgSource> = f
                .images
                .iter()
                .enumerate()
                .filter(|(_, im)| im.split == sp)
                .map(|(i, im)| BenchImgSource {
                    id: im.id.clone(),
                    file: im.file.rsplit('/').next().unwrap_or(&im.file).to_string(),
                    width: im.width,
                    height: im.height,
                    boxes: im.boxes.clone(),
                    seed: f.seed ^ (i as u64).wrapping_mul(0x9E37),
                    classes: f.ds.classes.len() as u32,
                    max_objects: f.max_objects,
                    small: f.small,
                })
                .collect();
            Ok((
                BenchSource::Dataset { dataset_id: f.ds.id.clone(), split: Some(sp) },
                f.ds.name.clone(),
                f.ds.classes.clone(),
                pool,
            ))
        }
        BenchSource::LabelJob { job_id } => {
            let job = jobs
                .iter()
                .find(|j| j.id == *job_id)
                .ok_or_else(|| ApiError::new("label_job_not_found", format!("No labeling job {job_id}")))?;
            let seed = hash(&job.id);
            let pool: Vec<BenchImgSource> = (0..job.counts.labeled)
                .map(|i| {
                    let s = seed ^ i.wrapping_mul(0x9E37);
                    let scene = Scene::generate(s, job.classes.len() as u32, 8, false);
                    BenchImgSource {
                        id: (i + 1).to_string(),
                        file: format!("frame_{:04}.jpg", i + 1),
                        width: 640,
                        height: 480,
                        boxes: scene.boxes(640, 480),
                        seed: s,
                        classes: job.classes.len() as u32,
                        max_objects: 8,
                        small: false,
                    }
                })
                .collect();
            Ok((
                BenchSource::LabelJob { job_id: job.id.clone() },
                job.name.clone(),
                job.classes.clone(),
                pool,
            ))
        }
    }
}
