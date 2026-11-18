//! The API's labeling records as the small results of [`crate::types::label`], and boxes between the stored
//! pixels of an image and the pixels of a view of it ([`PxBox`]).

use folo_api::{
    BBox, LabelAgent, LabelImage, LabelImagesAdded, LabelJob, LogLevel, LogLine, label_box_revision,
};

use crate::tools::short;
use crate::types::label::*;

/// Most class names a job summary lists.
const SUMMARY_CLASSES: usize = 100;
/// Most skipped files a result names.
const SKIPPED_FIRST: usize = 5;
/// Most boxes a view (or a saved edit) lists; the image draws them all.
pub const VIEW_BOXES: usize = 300;

/// The `[width, height]` of a view of a `width` × `height` image whose long side is at most `size`: the image's
/// own size when it is that small (never upscaled), else scaled so the long side is `size`.
pub fn frame(width: u32, height: u32, size: u32) -> [u32; 2] {
    let (width, height) = (width.max(1), height.max(1));
    let long = width.max(height);
    if long <= size {
        return [width, height];
    }
    let side = |v: u32| ((v as f64 * size as f64 / long as f64).round() as u32).clamp(1, size);
    if width >= height { [size, side(height)] } else { [side(width), size] }
}

/// The corners of a stored box in pixels of `frame`, unrounded (for drawing).
pub fn view_corners(b: &BBox, (width, height): (u32, u32), frame: [u32; 2]) -> [f32; 4] {
    let (sx, sy) = (frame[0] as f32 / width.max(1) as f32, frame[1] as f32 / height.max(1) as f32);
    [b.x1 * sx, b.y1 * sy, b.x2 * sx, b.y2 * sy]
}

/// A stored box as a [`PxBox`] of `frame`: corners rounded to whole pixels inside it.
pub fn to_view(b: &BBox, size: (u32, u32), frame: [u32; 2]) -> PxBox {
    let [x1, y1, x2, y2] = view_corners(b, size, frame);
    let px = |v: f32, side: u32| v.round().clamp(0.0, side as f32) as u32;
    [b.class, px(x1, frame[0]), px(y1, frame[1]), px(x2, frame[0]), px(y2, frame[1])]
}

/// A [`PxBox`] of `frame` as a stored box of a `width` × `height` image.
pub fn from_view(p: &PxBox, (width, height): (u32, u32), frame: [u32; 2]) -> BBox {
    let (sx, sy) = (width as f64 / frame[0].max(1) as f64, height as f64 / frame[1].max(1) as f64);
    let x = |v: u32| (v as f64 * sx) as f32;
    let y = |v: u32| (v as f64 * sy) as f32;
    BBox { class: p[0], x1: x(p[1]), y1: y(p[2]), x2: x(p[3]), y2: y(p[4]) }
}

/// The boxes of image `i` in pixels of `frame`.
pub(crate) fn view_boxes(i: &LabelImage, frame: [u32; 2]) -> Vec<PxBox> {
    i.boxes.iter().map(|b| to_view(b, (i.width, i.height), frame)).collect()
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

/// An image as `list_label_images` lists it; with `boxes`, its boxes in pixels of its default view.
pub(crate) fn image_summary(i: &LabelImage, boxes: bool) -> LabelImageSummary {
    let f = frame(i.width, i.height, DEFAULT_VIEW_SIZE);
    LabelImageSummary {
        id: i.id.clone(),
        file: short(&i.file, 128),
        status: i.status,
        num_boxes: i.boxes.len() as u32,
        source: i.source,
        reviewed: i.reviewed,
        error: i.error.as_deref().map(|e| short(e, 300)),
        size: boxes.then_some(f),
        revision: boxes.then(|| label_box_revision(&i.boxes)),
        boxes: boxes.then(|| view_boxes(i, f)),
    }
}

/// Image `i` in a view of `frame`, listing at most [`VIEW_BOXES`] boxes from number `from`.
pub(crate) fn image_view(i: &LabelImage, frame: [u32; 2], from: u32) -> LabelImageView {
    let from = from.max(1);
    let boxes = i
        .boxes
        .iter()
        .skip(from as usize - 1)
        .take(VIEW_BOXES)
        .map(|b| to_view(b, (i.width, i.height), frame))
        .collect();
    LabelImageView {
        id: i.id.clone(),
        file: short(&i.file, 128),
        size: frame,
        status: i.status,
        source: i.source,
        reviewed: i.reviewed,
        error: i.error.as_deref().map(|e| short(e, 300)),
        revision: label_box_revision(&i.boxes),
        num_boxes: i.boxes.len() as u32,
        from,
        boxes,
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

/// The text beside a view: the image and its frame, one line per listed box (`n class x1,y1,x2,y2`), how to list
/// the rest of a dense image's boxes, then the classes.
pub fn view_text(job_id: &str, v: &LabelImageView, classes: &[String], to_review: u64) -> String {
    let source = match v.source {
        Some(folo_api::LabelSource::Agent) => " by an agent",
        Some(folo_api::LabelSource::Manual) => " by hand",
        None => "",
    };
    let review = if v.reviewed { "reviewed" } else { "not reviewed" };
    let [w, h] = v.size;
    let mut text = format!(
        "{job_id} image {} ({}): {}{source}, {review}; {to_review} left to review.\nView {w}x{h} px, ticks every \
         {} px; boxes are [class, x1, y1, x2, y2] in these pixels; revision {}.",
        v.id,
        v.file,
        v.status.as_str(),
        super::view::tick_step(w.max(h)),
        v.revision
    );
    if let Some(e) = &v.error {
        text.push_str(&format!("\nLast agent call failed: {e}."));
    }
    if v.num_boxes == 0 {
        text.push_str("\nNo boxes.");
    } else {
        text.push_str("\nBoxes (n class x1,y1,x2,y2):");
        for (i, b) in v.boxes.iter().enumerate() {
            let name = classes.get(b[0] as usize).map_or("?", String::as_str);
            text.push_str(&format!("\n{} {name} {},{},{},{}", v.from as usize + i, b[1], b[2], b[3], b[4]));
        }
        let last = v.from as u64 + v.boxes.len() as u64 - 1;
        if v.from > 1 || last < v.num_boxes as u64 {
            text.push_str(&format!("\n(boxes {}-{last} of {}", v.from, v.num_boxes));
            if last < v.num_boxes as u64 {
                text.push_str(&format!("; view again with from: {} for the next", last + 1));
            }
            text.push(')');
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
    fn frames_scale_the_long_side_and_never_upscale() {
        assert_eq!(frame(1920, 1080, 1024), [1024, 576]);
        assert_eq!(frame(1080, 1920, 1024), [576, 1024]);
        assert_eq!(frame(640, 480, 1024), [640, 480]);
        assert_eq!(frame(5000, 3, 1024), [1024, 1]);
    }

    #[test]
    fn boxes_round_trip_through_view_pixels() {
        let size = (1920, 1080);
        let f = frame(1920, 1080, 1024);
        let b = BBox { class: 2, x1: 192.0, y1: 54.0, x2: 1919.4, y2: 1080.0 };
        let p = to_view(&b, size, f);
        assert_eq!(p, [2, 102, 29, 1024, 576]);
        let back = from_view(&p, size, f);
        assert_eq!((back.class, back.x1, back.y1, back.x2, back.y2), (2, 191.25, 54.375, 1920.0, 1080.0));
        // Outside the image is clamped, never wrapped.
        let out = to_view(&BBox { class: 0, x1: -5.0, y1: 0.0, x2: 5000.0, y2: 1.0 }, (100, 100), [100, 100]);
        assert_eq!(out, [0, 0, 0, 100, 1]);
    }
}
