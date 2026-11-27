//! Sequence accelerators for Agent Labeling:
//! - Jump to Next Unlabeled (`U`)
//! - Duplicate Selected Box (`Ctrl+D`, avoiding conflict with `D` Draw-mode toggle)
//! - Copy Previous Frame Boxes (`P`, with proportional cross-resolution scaling and deduplication)

use folo_ui::Context;

use super::editor::{MIN_BOX_PX, PendingImageSwitch, clamp_box_coords};
use super::state::{LabelImageFilter, SelectedBox};
use crate::api::{ApiError, BBox, LabelImage, LabelImageStatus};
use crate::app::FoloApp;
use crate::ui::{Tone, counted};

/// Tolerance (in image pixels) within which two same-class boxes count as exact duplicates
/// when copying boxes from a previous frame (absorbs 0.1 px rounding).
const DUP_TOLERANCE_PX: f32 = 0.15;

/// True when `im` still needs labels (`Pending` or `Failed`), matching `unlabeled_count`.
pub fn is_unlabeled_image(im: &LabelImage) -> bool {
    matches!(im.status, LabelImageStatus::Pending | LabelImageStatus::Failed)
}

/// Finds the next unlabeled image after `current_id` in the job sequence (wrapping around),
/// excluding `current_id` itself.
pub fn next_unlabeled_image_id(images: &[LabelImage], current_id: Option<&str>) -> Option<String> {
    if images.is_empty() {
        return None;
    }
    let start = current_id.and_then(|cid| images.iter().position(|im| im.id == cid));
    match start {
        Some(cur) => (1..images.len())
            .map(|off| &images[(cur + off) % images.len()])
            .find(|im| is_unlabeled_image(im))
            .map(|im| im.id.clone()),
        None => images.iter().find(|im| is_unlabeled_image(im)).map(|im| im.id.clone()),
    }
}

/// Finds the nearest preceding labeled image with at least one box before `current_id` in the job
/// sequence.
pub fn previous_labeled_frame<'a>(images: &'a [LabelImage], current_id: &str) -> Option<&'a LabelImage> {
    let cur = images.iter().position(|im| im.id == current_id)?;
    images[..cur].iter().rev().find(|im| im.status == LabelImageStatus::Labeled && !im.boxes.is_empty())
}

/// Computes a smart edge-aware offset duplicate of `orig` clamped inside `[0, img_w] × [0, img_h]`.
///
/// Prefers nudging down-right (`+delta`, `+delta`) where room exists, flipping to left (`-delta`)
/// or up (`-delta`) when near the right/bottom edges, and falling back to flush clamping when a
/// box spans nearly the entire axis so dimensions stay preserved.
pub fn duplicate_box_coords(orig: BBox, img_w: f32, img_h: f32) -> Option<BBox> {
    if !orig.x1.is_finite()
        || !orig.y1.is_finite()
        || !orig.x2.is_finite()
        || !orig.y2.is_finite()
        || !img_w.is_finite()
        || !img_h.is_finite()
        || img_w < MIN_BOX_PX
        || img_h < MIN_BOX_PX
    {
        return None;
    }
    let x1 = orig.x1.min(orig.x2).clamp(0.0, img_w);
    let y1 = orig.y1.min(orig.y2).clamp(0.0, img_h);
    let x2 = orig.x1.max(orig.x2).clamp(0.0, img_w);
    let y2 = orig.y1.max(orig.y2).clamp(0.0, img_h);
    let bw = x2 - x1;
    let bh = y2 - y1;
    if bw < MIN_BOX_PX || bh < MIN_BOX_PX {
        return None;
    }

    let delta = (img_w.min(img_h) * 0.025).clamp(8.0, 24.0);
    let shift_axis = |lo: f32, hi: f32, limit: f32| -> (f32, f32) {
        if hi + delta <= limit {
            (lo + delta, hi + delta)
        } else if lo - delta >= 0.0 {
            (lo - delta, hi - delta)
        } else if limit - hi >= 1.0 {
            let d = limit - hi;
            (lo + d, limit)
        } else if lo >= 1.0 {
            (0.0, hi - lo)
        } else {
            (lo, hi)
        }
    };

    let (nx1, nx2) = shift_axis(x1, x2, img_w);
    let (ny1, ny2) = shift_axis(y1, y2, img_h);
    let (cx1, cy1, cx2, cy2) = clamp_box_coords((nx1, ny1), (nx2, ny2), img_w, img_h)?;
    Some(BBox { class: orig.class, x1: cx1, y1: cy1, x2: cx2, y2: cy2 })
}

fn boxes_near_duplicate(a: BBox, b: BBox) -> bool {
    a.class == b.class
        && (a.x1 - b.x1).abs() <= DUP_TOLERANCE_PX
        && (a.y1 - b.y1).abs() <= DUP_TOLERANCE_PX
        && (a.x2 - b.x2).abs() <= DUP_TOLERANCE_PX
        && (a.y2 - b.y2).abs() <= DUP_TOLERANCE_PX
}

/// Scales `src_boxes` from `src_dims` to `dst_dims`, clamps them to image bounds, drops boxes
/// smaller than `MIN_BOX_PX` or with out-of-range classes, and skips exact duplicates already
/// present in `existing` (or earlier in `src_boxes`).
pub fn copy_scaled_boxes(
    src_boxes: &[BBox],
    src_dims: (u32, u32),
    dst_dims: (u32, u32),
    existing: &[BBox],
    num_classes: usize,
) -> Vec<BBox> {
    let sw = src_dims.0.max(1) as f32;
    let sh = src_dims.1.max(1) as f32;
    let dw = dst_dims.0.max(1) as f32;
    let dh = dst_dims.1.max(1) as f32;
    let sx = if src_dims.0 == dst_dims.0 { 1.0 } else { dw / sw };
    let sy = if src_dims.1 == dst_dims.1 { 1.0 } else { dh / sh };

    let mut added: Vec<BBox> = Vec::new();
    for &b in src_boxes {
        if num_classes > 0 && (b.class as usize) >= num_classes {
            continue;
        }
        let Some((x1, y1, x2, y2)) = clamp_box_coords((b.x1 * sx, b.y1 * sy), (b.x2 * sx, b.y2 * sy), dw, dh)
        else {
            continue;
        };
        let cand = BBox { class: b.class, x1, y1, x2, y2 };
        if existing.iter().any(|&e| boxes_near_duplicate(e, cand))
            || added.iter().any(|&a| boxes_near_duplicate(a, cand))
        {
            continue;
        }
        added.push(cand);
    }
    added
}

impl FoloApp {
    /// Jumps to the next unlabeled (`Pending` or `Failed`) image in `job_id`'s sequence (`U`),
    /// guarding unsaved edits via `PendingImageSwitch` and giving clear status feedback when all
    /// images are already labeled.
    pub fn request_jump_next_unlabeled(&mut self, job_id: &str, cx: &mut Context<Self>) {
        let Some(all) = self.label.images.get(job_id).and_then(|l| l.ready()).cloned() else {
            return;
        };
        if all.is_empty() {
            return;
        }
        let active_id = self.active_label_image_id(job_id);
        if let Some(target_id) = next_unlabeled_image_id(&all, active_id.as_deref()) {
            let shown = self.current_filtered_label_images(job_id);
            let clear_filters = !shown.iter().any(|im| im.id == target_id);
            if self.label.editor.is_dirty() {
                self.label.editor.pending_switch =
                    Some(PendingImageSwitch::JumpToImage { image_id: target_id, clear_filters });
                cx.notify();
                return;
            }
            self.jump_to_label_image(job_id, target_id, clear_filters, cx);
            return;
        }

        // No other unlabeled image exists in the job; give clear status feedback.
        if let Some(cur_im) = active_id.as_deref().and_then(|cid| all.iter().find(|im| im.id == cid))
            && is_unlabeled_image(cur_im)
        {
            self.toast(
                Tone::Info,
                "Already on the only unlabeled image",
                Some(format!("{} is the last remaining unlabeled image.", cur_im.file).into()),
                cx,
            );
            return;
        }
        if all.iter().any(|im| im.status == LabelImageStatus::Labeling) {
            self.toast(
                Tone::Info,
                "No pending images",
                Some("Remaining images are currently being labeled by the agent.".into()),
                cx,
            );
            return;
        }
        self.toast(
            Tone::Success,
            "All images are labeled",
            Some(
                format!("All {} in this job have labels.", counted(all.len() as u64, "image", "images"))
                    .into(),
            ),
            cx,
        );
    }

    pub(super) fn jump_to_label_image(
        &mut self,
        job_id: &str,
        image_id: String,
        clear_filters: bool,
        cx: &mut Context<Self>,
    ) {
        if clear_filters {
            self.label.img_filter = LabelImageFilter::All;
            self.label.class_filter = None;
        }
        let target_file = self
            .label
            .images
            .get(job_id)
            .and_then(|l| l.ready())
            .and_then(|list| list.iter().find(|im| im.id == image_id))
            .map(|im| im.file.clone());
        self.select_label_image(image_id, cx);
        if clear_filters && let Some(file) = target_file {
            self.toast(
                Tone::Info,
                "Switched filter to All",
                Some(format!("Jumped to unlabeled image {file}.").into()),
                cx,
            );
        }
    }

    /// Duplicates the currently selected box on `image_id` with smart edge-aware offset (`Ctrl+D`),
    /// selects the duplicate, pushes to the undo stack, and marks the draft dirty.
    pub fn duplicate_selected_label_box(&mut self, job_id: &str, image_id: &str, cx: &mut Context<Self>) {
        if self.label.mutating {
            return;
        }
        let Some(sel) = self.label.selected_box.clone() else { return };
        if sel.image_id != image_id {
            self.label.selected_box = None;
            cx.notify();
            return;
        }
        let Some(im) = self.ensure_label_editor_synced(job_id, image_id) else {
            return;
        };
        if im.status == LabelImageStatus::Labeling {
            self.label.review_error =
                Some(ApiError::new("label_job_active", "The agent is labeling this image right now."));
            cx.notify();
            return;
        }
        if sel.index >= self.label.editor.draft.len() || self.label.editor.draft_revision() != sel.revision {
            self.label.selected_box = None;
            self.label.hover_box = None;
            self.label.review_error = Some(ApiError::new(
                "label_boxes_changed",
                "This image's boxes changed; reselect the box before duplicating.",
            ));
            cx.notify();
            return;
        }
        let (w, h) = (im.width.max(1) as f32, im.height.max(1) as f32);
        if let Some(new_idx) = self.label.editor.duplicate_box(sel.index, w, h) {
            let new_rev = self.label.editor.draft_revision();
            self.label.selected_box =
                Some(SelectedBox { image_id: image_id.to_string(), index: new_idx, revision: new_rev });
            self.label.hover_box = None;
            self.label.review_error = None;
            cx.notify();
        }
    }

    /// Copies boxes from the nearest preceding labeled image onto `image_id` (`P`), scaling
    /// coordinates proportionally if frame dimensions differ, clamping to bounds, avoiding exact
    /// duplicates, and committing all copied boxes as one atomic undo step.
    pub fn copy_prev_frame_label_boxes(&mut self, job_id: &str, image_id: &str, cx: &mut Context<Self>) {
        if self.label.mutating {
            return;
        }
        let Some(im) = self.ensure_label_editor_synced(job_id, image_id) else {
            return;
        };
        if im.status == LabelImageStatus::Labeling {
            self.label.review_error =
                Some(ApiError::new("label_job_active", "The agent is labeling this image right now."));
            cx.notify();
            return;
        }
        let Some(all) = self.label.images.get(job_id).and_then(|l| l.ready()).cloned() else {
            return;
        };
        let Some(cur_idx) = all.iter().position(|i| i.id == image_id) else {
            return;
        };
        if cur_idx == 0 {
            self.toast(
                Tone::Info,
                "No previous frame",
                Some("This is the first image in the job sequence.".into()),
                cx,
            );
            return;
        }
        let Some(prev) = previous_labeled_frame(&all, image_id) else {
            self.toast(
                Tone::Info,
                "No previous labeled boxes",
                Some("No earlier labeled image in this sequence has boxes to copy.".into()),
                cx,
            );
            return;
        };
        let n_classes = self.label_job_by_id(job_id).map_or(0, |j| j.classes.len());
        let added = copy_scaled_boxes(
            &prev.boxes,
            (prev.width, prev.height),
            (im.width, im.height),
            &self.label.editor.draft,
            n_classes,
        );
        if added.is_empty() {
            self.toast(
                Tone::Info,
                "Boxes already present",
                Some(format!("All boxes from {} are already on this image.", prev.file).into()),
                cx,
            );
            return;
        }
        let scaled = prev.width != im.width || prev.height != im.height;
        let count = self.label.editor.add_boxes_atomic(added);
        if count > 0 {
            let new_idx = self.label.editor.draft.len() - 1;
            let new_rev = self.label.editor.draft_revision();
            self.label.selected_box =
                Some(SelectedBox { image_id: image_id.to_string(), index: new_idx, revision: new_rev });
            self.label.hover_box = None;
            self.label.review_error = None;
            let detail = if scaled {
                format!("From {} ({}×{} → {}×{})", prev.file, prev.width, prev.height, im.width, im.height)
            } else {
                format!("From {}", prev.file)
            };
            self.toast(
                Tone::Success,
                format!("Copied {}", counted(count as u64, "box", "boxes")),
                Some(detail.into()),
                cx,
            );
            cx.notify();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::LabelSource;

    fn img(id: &str, status: LabelImageStatus, w: u32, h: u32, boxes: Vec<BBox>) -> LabelImage {
        LabelImage {
            id: id.into(),
            file: format!("{id}.jpg"),
            width: w,
            height: h,
            status,
            boxes,
            source: (status == LabelImageStatus::Labeled).then_some(LabelSource::Agent),
            error: None,
            reviewed: false,
        }
    }

    #[test]
    fn duplicate_box_coords_offsets_cleanly_and_flips_at_image_edges() {
        // Interior box shifts down-right by delta = (480 * 0.025) = 12 px.
        let b = BBox { class: 1, x1: 100.0, y1: 80.0, x2: 200.0, y2: 160.0 };
        let dup = duplicate_box_coords(b, 640.0, 480.0).unwrap();
        assert_eq!(dup, BBox { class: 1, x1: 112.0, y1: 92.0, x2: 212.0, y2: 172.0 });

        // Near bottom-right corner flips to up-left so box stays inside [0, 640] x [0, 480].
        let corner = BBox { class: 2, x1: 535.0, y1: 395.0, x2: 635.0, y2: 475.0 };
        let dup_corner = duplicate_box_coords(corner, 640.0, 480.0).unwrap();
        assert_eq!(dup_corner, BBox { class: 2, x1: 523.0, y1: 383.0, x2: 623.0, y2: 463.0 });
        assert!(dup_corner.x1 >= 0.0 && dup_corner.x2 <= 640.0);
        assert!(dup_corner.y1 >= 0.0 && dup_corner.y2 <= 480.0);

        // Full-frame box preserves full dimensions within bounds without panicking.
        let full = BBox { class: 0, x1: 0.0, y1: 0.0, x2: 640.0, y2: 480.0 };
        let dup_full = duplicate_box_coords(full, 640.0, 480.0).unwrap();
        assert_eq!(dup_full, full);

        // Degenerate / non-finite boxes return None.
        assert_eq!(
            duplicate_box_coords(BBox { class: 0, x1: f32::NAN, y1: 0.0, x2: 50.0, y2: 50.0 }, 640.0, 480.0),
            None
        );
        assert_eq!(
            duplicate_box_coords(BBox { class: 0, x1: 10.0, y1: 10.0, x2: 12.0, y2: 50.0 }, 640.0, 480.0),
            None
        );
    }

    #[test]
    fn copy_scaled_boxes_handles_cross_resolution_clamping_and_dedup() {
        let src = vec![
            BBox { class: 0, x1: 64.0, y1: 48.0, x2: 320.0, y2: 240.0 },
            // Tiny box that shrinks below MIN_BOX_PX on 320x240 target.
            BBox { class: 1, x1: 10.0, y1: 10.0, x2: 16.0, y2: 16.0 },
            // Out-of-range class (dropped when num_classes = 2).
            BBox { class: 5, x1: 100.0, y1: 100.0, x2: 200.0, y2: 200.0 },
            // Duplicate of box 0 inside src.
            BBox { class: 0, x1: 64.0, y1: 48.0, x2: 320.0, y2: 240.0 },
        ];
        // Scale 640x480 -> 1280x960 (2x both axes).
        let up = copy_scaled_boxes(&src, (640, 480), (1280, 960), &[], 2);
        assert_eq!(up.len(), 2);
        assert_eq!(up[0], BBox { class: 0, x1: 128.0, y1: 96.0, x2: 640.0, y2: 480.0 });
        assert_eq!(up[1], BBox { class: 1, x1: 20.0, y1: 20.0, x2: 32.0, y2: 32.0 });

        // Scale down 640x480 -> 320x240: tiny box becomes 3x3 (< MIN_BOX_PX) and is dropped.
        let down = copy_scaled_boxes(&src, (640, 480), (320, 240), &[], 2);
        assert_eq!(down, vec![BBox { class: 0, x1: 32.0, y1: 24.0, x2: 160.0, y2: 120.0 }]);

        // Existing exact duplicate is skipped.
        let deduped = copy_scaled_boxes(&src, (640, 480), (320, 240), &down, 2);
        assert!(deduped.is_empty());
    }

    #[test]
    fn next_unlabeled_and_previous_labeled_frame_helpers() {
        let b0 = BBox { class: 0, x1: 10.0, y1: 10.0, x2: 80.0, y2: 80.0 };
        let list = vec![
            img("i0", LabelImageStatus::Pending, 640, 480, vec![]),
            img("i1", LabelImageStatus::Labeled, 640, 480, vec![b0]),
            img("i2", LabelImageStatus::Labeled, 640, 480, vec![]),
            img("i3", LabelImageStatus::Failed, 640, 480, vec![]),
            img("i4", LabelImageStatus::Pending, 640, 480, vec![]),
        ];
        assert_eq!(next_unlabeled_image_id(&list, Some("i1")).as_deref(), Some("i3"));
        assert_eq!(next_unlabeled_image_id(&list, Some("i3")).as_deref(), Some("i4"));
        // Wraps around from i4 back to i0.
        assert_eq!(next_unlabeled_image_id(&list, Some("i4")).as_deref(), Some("i0"));

        // Nearest preceding labeled image with boxes skips empty labeled i2.
        assert_eq!(previous_labeled_frame(&list, "i3").map(|im| im.id.as_str()), Some("i1"));
        assert_eq!(previous_labeled_frame(&list, "i0"), None);
        assert_eq!(previous_labeled_frame(&list, "i1"), None);

        // When all images are labeled, next_unlabeled returns None.
        let all_done = vec![img("a", LabelImageStatus::Labeled, 640, 480, vec![b0])];
        assert_eq!(next_unlabeled_image_id(&all_done, Some("a")), None);
    }
}
