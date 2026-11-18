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

/// The text beside a view, at most `budget` bytes: the image and its frame, the classes, one line per listed box
/// (`n class x1,y1,x2,y2`) and how to list the rest. The boxes are cut to the budget, never the paging line or the
/// classes: a dense image's text once lost both to a cap on its tail.
pub fn view_text(
    job_id: &str,
    v: &LabelImageView,
    classes: &[String],
    to_review: u64,
    budget: usize,
) -> String {
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
    let classes_line = class_line(classes, budget / 4);
    // Box lines name the class when every name is short, else give its index (the line stays small).
    let by_name = classes.iter().all(|c| c.len() <= SHORT_CLASS);
    let mut lines = String::new();
    let mut listed = 0u64;
    // Paging past the end first: boxes cleared since paging started are no plain "No boxes."
    if v.num_boxes == 0 && v.from <= 1 {
        lines.push_str("\nNo boxes.");
    } else if v.boxes.is_empty() {
        lines.push_str(&format!(
            "\nNo boxes from number {}: the image has {} now (revision {}). If you were paging another revision, \
             its boxes changed; view again with from: 1.",
            v.from, v.num_boxes, v.revision
        ));
    } else {
        // Room for the paging line, which is under 120 bytes.
        let room = budget.saturating_sub(text.len() + classes_line.len() + 120);
        lines.push_str(if by_name {
            "\nBoxes (n class x1,y1,x2,y2):"
        } else {
            "\nBoxes (n class-index x1,y1,x2,y2):"
        });
        for (i, b) in v.boxes.iter().enumerate() {
            let class = match classes.get(b[0] as usize) {
                Some(name) if by_name => name.clone(),
                _ => b[0].to_string(),
            };
            let line = format!("\n{} {class} {},{},{},{}", v.from as usize + i, b[1], b[2], b[3], b[4]);
            if lines.len() + line.len() > room {
                break;
            }
            lines.push_str(&line);
            listed += 1;
        }
    }
    text.push_str(&lines);
    if listed > 0 {
        let last = v.from as u64 + listed - 1;
        if v.from > 1 || last < v.num_boxes as u64 {
            text.push_str(&format!("\n(boxes {}-{last} of {}", v.from, v.num_boxes));
            if last < v.num_boxes as u64 {
                text.push_str(&format!("; view again with from: {} for the next", last + 1));
            }
            text.push(')');
        }
    }
    text.push_str(&classes_line);
    text
}

/// Longest class name a box line of [`view_text`] spells out.
const SHORT_CLASS: usize = 12;

/// `\nClasses: 0 car, 1 person, ...` in at most `budget` bytes: the names that fit, then how many are left out.
fn class_line(classes: &[String], budget: usize) -> String {
    let mut line = String::from("\nClasses:");
    for (i, c) in classes.iter().enumerate() {
        let entry = format!("{} {i} {}", if i == 0 { "" } else { "," }, short(c, 64));
        if line.len() + entry.len() + 60 > budget {
            line.push_str(&format!(" ... and {} more (the `classes` of the result)", classes.len() - i));
            break;
        }
        line.push_str(&entry);
    }
    line
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

    fn view(num_boxes: u32, from: u32) -> LabelImageView {
        let boxes = (from..=num_boxes).take(VIEW_BOXES).map(|k| [0, k, k, k + 10, k + 10]).collect();
        LabelImageView {
            id: "7".into(),
            file: "a.jpg".into(),
            size: [1024, 1024],
            status: folo_api::LabelImageStatus::Labeled,
            source: None,
            reviewed: false,
            error: None,
            revision: "n-x".into(),
            num_boxes,
            from,
            boxes,
        }
    }

    #[test]
    fn view_text_keeps_the_paging_line_and_the_classes_within_its_budget() {
        let classes: Vec<String> = (0..500).map(|i| format!("a-rather-long-class-name-{i}")).collect();
        let t = view_text("lj_1", &view(1000, 1), &classes, 3, 4000);
        assert!(t.len() <= 4000, "{}", t.len());
        assert!(t.contains("; view again with from: "), "{t}");
        assert!(
            t.contains("\nClasses: 0 a-rather-long-class-name-0,") && t.contains("more (the `classes`"),
            "{t}"
        );
        // Long names: box lines give the class index.
        assert!(t.contains("\n1 0 1,1,11,11\n"), "{t}");
        let next: u32 = t.split("from: ").nth(1).unwrap().split(' ').next().unwrap().parse().unwrap();
        assert!(t.contains(&format!("\n{} 0 ", next - 1)) && !t.contains(&format!("\n{next} 0 ")), "{t}");

        let short = view_text("lj_1", &view(2, 1), &["car".into()], 0, 4000);
        assert!(short.ends_with("\n2 car 2,2,12,12\nClasses: 0 car"), "{short}");
    }

    #[test]
    fn a_view_past_the_last_box_says_to_start_again() {
        let t = view_text("lj_1", &view(5, 9), &["car".into()], 0, 16_000);
        assert!(t.contains("No boxes from number 9: the image has 5 now (revision n-x)"), "{t}");
        assert!(t.contains("view again with from: 1.") && !t.contains("(boxes"), "{t}");
        // All of them cleared since paging started.
        let t = view_text("lj_1", &view(0, 301), &["car".into()], 0, 16_000);
        assert!(t.contains("No boxes from number 301: the image has 0 now"), "{t}");
    }
}
