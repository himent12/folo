//! The API's labeling records as the small results of [`crate::types::label`], and boxes between pixels and the
//! 0..1000 units of a [`RelBox`].

use folo_api::{BBox, LabelAgent, LabelImage, LabelImagesAdded, LabelJob, LogLevel, LogLine};

use crate::tools::short;
use crate::types::label::*;

/// Most class names a job summary lists.
const SUMMARY_CLASSES: usize = 100;
/// Most skipped files a result names.
const SKIPPED_FIRST: usize = 5;

/// A pixel box of a `width` × `height` image as `[class, x1, y1, x2, y2]` in 0..1000.
pub fn to_rel(b: &BBox, width: u32, height: u32) -> RelBox {
    let rel =
        |v: f32, side: u32| ((v as f64 / side.max(1) as f64 * 1000.0).round().clamp(0.0, 1000.0)) as u32;
    [b.class, rel(b.x1, width), rel(b.y1, height), rel(b.x2, width), rel(b.y2, height)]
}

/// A [`RelBox`] as a pixel box of a `width` × `height` image.
pub fn from_rel(r: &RelBox, width: u32, height: u32) -> BBox {
    let px = |v: u32, side: u32| (v.min(1000) as f64 / 1000.0 * side as f64) as f32;
    BBox { class: r[0], x1: px(r[1], width), y1: px(r[2], height), x2: px(r[3], width), y2: px(r[4], height) }
}

pub(crate) fn agent(a: LabelAgent) -> LabelAgentSummary {
    LabelAgentSummary {
        id: a.id,
        available: a.available,
        version: a.version,
        models: a.models.into_iter().map(|m| m.id).collect(),
        default_model: a.default_model,
        default_effort: a.default_effort,
        hint: a.hint,
    }
}

pub(crate) fn job(j: LabelJob) -> LabelJobSummary {
    LabelJobSummary {
        id: j.id,
        name: short(&j.name, 128),
        status: j.status,
        classes: j.classes.iter().take(SUMMARY_CLASSES).map(|c| short(c, 64)).collect(),
        counts: j.counts,
        dataset_id: j.dataset_id,
    }
}

/// What one or more adds to job `job_id` did; `images` is what the job has after the last.
pub(crate) fn added(job_id: &str, adds: &[LabelImagesAdded], images: u64) -> LabelImagesAddedResult {
    let skipped = adds.iter().flat_map(|a| &a.skipped);
    LabelImagesAddedResult {
        job_id: job_id.to_string(),
        images,
        added: adds.iter().map(|a| a.added.len() as u64).sum(),
        skipped: skipped.clone().count() as u64,
        skipped_first: skipped
            .take(SKIPPED_FIRST)
            .map(|s| folo_api::SkippedImage { file: short(&s.file, 200), reason: short(&s.reason, 200) })
            .collect(),
    }
}

pub(crate) fn image_summary(i: &LabelImage, include_boxes: bool) -> LabelImageSummary {
    LabelImageSummary {
        id: i.id.clone(),
        file: short(&i.file, 128),
        status: i.status,
        num_boxes: i.boxes.len() as u32,
        source: i.source,
        reviewed: i.reviewed,
        error: i.error.as_deref().map(|e| short(e, 300)),
        boxes: include_boxes.then(|| rel_boxes(i)),
    }
}

pub(crate) fn rel_boxes(i: &LabelImage) -> Vec<RelBox> {
    i.boxes.iter().map(|b| to_rel(b, i.width, i.height)).collect()
}

pub(crate) fn image_view(i: &LabelImage) -> LabelImageView {
    LabelImageView {
        id: i.id.clone(),
        file: short(&i.file, 128),
        width: i.width,
        height: i.height,
        status: i.status,
        source: i.source,
        reviewed: i.reviewed,
        error: i.error.as_deref().map(|e| short(e, 300)),
        boxes: rel_boxes(i),
    }
}

/// A log line as `label_status` lists it: the message, prefixed with `warn: ` or `error: `.
pub(crate) fn log_line(l: &LogLine) -> String {
    let message = short(&l.message, 300);
    match l.level {
        LogLevel::Warn => format!("warn: {message}"),
        LogLevel::Error => format!("error: {message}"),
        _ => message,
    }
}

/// The text beside a view: the image, then one line per box (`n class x1,y1,x2,y2`), then the classes.
pub fn view_text(job_id: &str, v: &LabelImageView, classes: &[String], to_review: u64) -> String {
    let source = match v.source {
        Some(folo_api::LabelSource::Agent) => " by an agent",
        Some(folo_api::LabelSource::Manual) => " by hand",
        None => "",
    };
    let review = if v.reviewed { "reviewed" } else { "not reviewed" };
    let mut text = format!(
        "{job_id} image {} ({}, {}x{}): {}{source}, {review}; {} left to review.",
        v.id,
        v.file,
        v.width,
        v.height,
        v.status.as_str(),
        to_review
    );
    if let Some(e) = &v.error {
        text.push_str(&format!(" Last agent call failed: {e}."));
    }
    if v.boxes.is_empty() {
        text.push_str("\nNo boxes.");
    } else {
        text.push_str("\nBoxes (n class x1,y1,x2,y2 in 0..1000 of width/height):");
        for (n, b) in v.boxes.iter().enumerate() {
            let name = classes.get(b[0] as usize).map_or("?", String::as_str);
            text.push_str(&format!("\n{} {name} {},{},{},{}", n + 1, b[1], b[2], b[3], b[4]));
        }
    }
    let classes: Vec<String> = classes.iter().enumerate().map(|(i, c)| format!("{i} {c}")).collect();
    text.push_str(&format!("\nClasses: {}", classes.join(", ")));
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boxes_round_trip_through_relative_units() {
        let b = BBox { class: 2, x1: 192.0, y1: 54.0, x2: 1919.4, y2: 1080.0 };
        let r = to_rel(&b, 1920, 1080);
        assert_eq!(r, [2, 100, 50, 1000, 1000]);
        let back = from_rel(&r, 1920, 1080);
        assert_eq!((back.class, back.x1, back.y1, back.x2, back.y2), (2, 192.0, 54.0, 1920.0, 1080.0));
        // Out of range values are clamped, never wrapped.
        assert_eq!(
            to_rel(&BBox { class: 0, x1: -5.0, y1: 0.0, x2: 5000.0, y2: 1.0 }, 100, 100)[1..3],
            [0, 0]
        );
        assert_eq!(from_rel(&[0, 0, 0, 1500, 10], 200, 100).x2, 200.0);
    }
}
