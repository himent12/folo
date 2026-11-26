//! Validation sample selection, tile tooltips and operating-threshold helpers for Playground.

use crate::api::{ImageInfo, ModelMetrics, Split};
use crate::ui::{counted, operating_conf};

/// Picks up to `cap` validation images for the Playground sample strip: annotated images first, then
/// background (0-box) validation images so false-positive checks and background-only val splits still
/// offer clickable samples.
pub(crate) fn pick_val_samples(images: &[ImageInfo], cap: usize) -> Vec<ImageInfo> {
    let mut out: Vec<ImageInfo> =
        images.iter().filter(|m| m.split == Split::Val && !m.boxes.is_empty()).take(cap).cloned().collect();
    if out.len() < cap {
        let need = cap - out.len();
        out.extend(images.iter().filter(|m| m.split == Split::Val && m.boxes.is_empty()).take(need).cloned());
    }
    out
}

/// Tooltip copy for a validation sample tile, calling out background (0-box) images explicitly.
pub(crate) fn sample_tile_tip(m: &ImageInfo) -> String {
    let file = crate::screens::datasets::file_name(&m.file);
    let name = if file.is_empty() { m.id.as_str() } else { file };
    if m.boxes.is_empty() {
        format!("{name} · background · 0 boxes")
    } else {
        format!("{name} · {}", counted(m.boxes.len() as u64, "box", "boxes"))
    }
}

/// The confidence Playground offers as "Use … (best F1)": the model's best-F1 threshold, unless the
/// model found nothing (precision and recall 0), where that threshold is only the end of the search
/// grid and using it would hide every detection.
pub(crate) fn offered_conf(m: &ModelMetrics) -> Option<f32> {
    operating_conf(m.best_f1_conf, m.precision, m.recall)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn val_img(id: &str, file: &str, split: Split, n_boxes: usize) -> ImageInfo {
        ImageInfo {
            id: id.into(),
            file: file.into(),
            split,
            width: 640,
            height: 480,
            boxes: (0..n_boxes)
                .map(|_| crate::api::BBox { class: 0, x1: 10., y1: 10., x2: 50., y2: 50. })
                .collect(),
        }
    }

    #[test]
    fn best_f1_conf_is_offered_only_when_the_model_found_something() {
        let m = |precision: f32, recall: f32| ModelMetrics {
            map50: 0.,
            map50_95: 0.,
            precision,
            recall,
            best_f1_conf: Some(1.0),
        };
        // A model that learned nothing yet: its "best" threshold is 1.00, which shows no boxes.
        assert_eq!(offered_conf(&m(0., 0.)), None);
        assert_eq!(offered_conf(&m(0.4, 0.)), Some(1.0));
        assert_eq!(offered_conf(&m(0.8, 0.6)), Some(1.0));
        assert_eq!(offered_conf(&ModelMetrics { best_f1_conf: None, ..m(0.8, 0.6) }), None);
    }

    #[test]
    fn pick_val_samples_prefers_annotated_then_fills_with_background_val_images() {
        let images = vec![
            val_img("t1", "train/a.jpg", Split::Train, 2),
            val_img("bg1", "val/bg_1.jpg", Split::Val, 0),
            val_img("ann1", "val/ann_1.jpg", Split::Val, 3),
            val_img("bg2", "val/bg_2.jpg", Split::Val, 0),
            val_img("ann2", "val/ann_2.jpg", Split::Val, 1),
            val_img("bg3", "val/bg_3.jpg", Split::Val, 0),
        ];
        let picked = pick_val_samples(&images, 4);
        let ids: Vec<&str> = picked.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(ids, vec!["ann1", "ann2", "bg1", "bg2"]);
        assert_eq!(sample_tile_tip(&picked[0]), "ann_1.jpg · 3 boxes");
        assert_eq!(sample_tile_tip(&picked[1]), "ann_2.jpg · 1 box");
        assert_eq!(sample_tile_tip(&picked[2]), "bg_1.jpg · background · 0 boxes");
    }

    #[test]
    fn pick_val_samples_handles_background_only_val_split() {
        let images = vec![
            val_img("t1", "train/a.jpg", Split::Train, 4),
            val_img("bg1", "val/empty_a.jpg", Split::Val, 0),
            val_img("bg2", "val/empty_b.jpg", Split::Val, 0),
        ];
        let picked = pick_val_samples(&images, 4);
        let ids: Vec<&str> = picked.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(ids, vec!["bg1", "bg2"]);
    }
}
