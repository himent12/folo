//! Interactive bounding-box editor state machine for Agent Labeling.
//!
//! Holds the local working copy (`draft`) over the server's last known boxes (`base`), bounded
//! undo/redo history, active draw tool/class, live click-drag preview state, and unsaved-changes
//! guards when stepping between images or receiving concurrent SSE updates.

use crate::api::{BBox, LabelImage, label_box_revision};
use crate::app::Route;

use super::state::LabelImageFilter;

/// Maximum undo snapshots kept per open image.
const MAX_HISTORY: usize = 100;

/// Minimum width and height (in image pixels) for a drawn bounding box to be committed.
pub const MIN_BOX_PX: f32 = 4.0;

/// Minimum pointer drag distance (in logical screen pixels) before a press counts as a box draw
/// rather than a click.
pub const DRAG_THRESHOLD_PX: f32 = 3.0;

/// Canvas pointer tool for the labeling review stage.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum EditTool {
    /// Click an existing box to select it; click-drag on empty image space to draw a new box.
    #[default]
    Select,
    /// Click-drag anywhere on the image (even overlapping existing boxes) to draw a new box.
    Draw,
}

/// Active click-drag box creation on the review canvas.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoxDraft {
    pub start_img: (f32, f32),
    pub cur_img: (f32, f32),
    pub start_px: (f32, f32),
    pub cur_px: (f32, f32),
    pub class: u32,
    /// Becomes true once pointer travel exceeds [`DRAG_THRESHOLD_PX`].
    pub moved: bool,
}

/// Deferred navigation/filter action awaiting user choice when unsaved box edits exist.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum PendingImageSwitch {
    SelectImage(String),
    Step(i32),
    SetFilter(LabelImageFilter),
    SetClassFilter(Option<u32>),
    Navigate(Route),
    Back,
    Forward,
    CloseWindow,
}

/// Deferred route navigation executed on the next render frame (when `&mut Window` is available).
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum PendingRouteNav {
    Navigate(Route),
    Back,
    Forward,
    CloseWindow,
}

/// Exact structural equality for bounding-box lists (using bit patterns for floats so NaN or
/// signed-zero quirks never cause false dirty states).
pub fn same_boxes(a: &[BBox], b: &[BBox]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| {
            x.class == y.class
                && x.x1.to_bits() == y.x1.to_bits()
                && x.y1.to_bits() == y.y1.to_bits()
                && x.x2.to_bits() == y.x2.to_bits()
                && x.y2.to_bits() == y.y2.to_bits()
        })
}

/// Normalizes and clamps two image-space corner points to `[0, img_w] × [0, img_h]`.
/// Returns `None` if any coordinate/dimension is non-finite or the resulting box is smaller than
/// [`MIN_BOX_PX`] in either axis.
pub fn clamp_box_coords(
    p0: (f32, f32),
    p1: (f32, f32),
    img_w: f32,
    img_h: f32,
) -> Option<(f32, f32, f32, f32)> {
    if !p0.0.is_finite()
        || !p0.1.is_finite()
        || !p1.0.is_finite()
        || !p1.1.is_finite()
        || !img_w.is_finite()
        || !img_h.is_finite()
        || img_w < MIN_BOX_PX
        || img_h < MIN_BOX_PX
    {
        return None;
    }
    let x1 = p0.0.min(p1.0).clamp(0.0, img_w);
    let y1 = p0.1.min(p1.1).clamp(0.0, img_h);
    let x2 = p0.0.max(p1.0).clamp(0.0, img_w);
    let y2 = p0.1.max(p1.1).clamp(0.0, img_h);
    if (x2 - x1) < MIN_BOX_PX || (y2 - y1) < MIN_BOX_PX {
        return None;
    }
    // Round to 0.1 px precision for clean serialized coordinates matching Studio.
    let r = |v: f32| (v * 10.0).round() / 10.0;
    let (x1, y1, x2, y2) = (r(x1), r(y1), r(x2), r(y2));
    if (x2 - x1) < MIN_BOX_PX || (y2 - y1) < MIN_BOX_PX {
        return None;
    }
    Some((x1, y1, x2, y2))
}

/// Local draft, undo/redo stacks, active drawing class/tool, and live drag preview for the active
/// labeling image.
#[derive(Clone, Debug, Default)]
pub struct LabelEditor {
    pub job_id: Option<String>,
    pub image_id: Option<String>,
    /// Last known server boxes for `(job_id, image_id)`.
    pub base: Vec<BBox>,
    /// Server box revision this draft was originally edited against. Kept unchanged across SSE
    /// updates while dirty so a plain Save (`Ctrl+S`) cannot silently overwrite another client's
    /// concurrent edits; only refreshed on clean sync, Discard, Save confirmation, or explicit overwrite.
    pub expect_rev: String,
    /// Working boxes currently rendered and edited.
    pub draft: Vec<BBox>,
    /// Past snapshots for `Ctrl+Z` (most recent last).
    pub undo: Vec<Vec<BBox>>,
    /// Future snapshots for `Ctrl+Shift+Z` / `Ctrl+Y` (most recent last).
    pub redo: Vec<Vec<BBox>>,
    pub tool: EditTool,
    /// Class index applied to newly drawn boxes.
    pub draw_class: u32,
    /// In-progress click-drag rectangle on the stage.
    pub drawing: Option<BoxDraft>,
    /// True if an SSE update or reload changed the image's server boxes while `draft` was dirty.
    pub changed_elsewhere: bool,
    /// Deferred image/filter switch waiting for the user to Save, Discard, or Stay.
    pub pending_switch: Option<PendingImageSwitch>,
}

impl LabelEditor {
    /// Clears all draft, history, and drag state (on job switch, route exit, or backend reset).
    pub fn reset(&mut self) {
        self.job_id = None;
        self.image_id = None;
        self.base.clear();
        self.expect_rev.clear();
        self.draft.clear();
        self.undo.clear();
        self.redo.clear();
        self.tool = EditTool::Select;
        self.drawing = None;
        self.changed_elsewhere = false;
        self.pending_switch = None;
    }

    /// True when bound to `(job_id, image_id)`.
    pub fn is_for(&self, job_id: &str, image_id: &str) -> bool {
        self.job_id.as_deref() == Some(job_id) && self.image_id.as_deref() == Some(image_id)
    }

    /// True when the local working copy differs from the server's `base`.
    pub fn is_dirty(&self) -> bool {
        self.image_id.is_some() && !same_boxes(&self.draft, &self.base)
    }

    /// True when bound to `(job_id, image_id)` and holding unsaved edits.
    pub fn is_dirty_for(&self, job_id: &str, image_id: &str) -> bool {
        self.is_for(job_id, image_id) && !same_boxes(&self.draft, &self.base)
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    /// The revision of `base`, sent as `UpdateLabelImage::expect` so stale overwrites are caught.
    pub fn base_revision(&self) -> String {
        label_box_revision(&self.base)
    }

    /// The revision of the current `draft`, used to bind box selection safely.
    pub fn draft_revision(&self) -> String {
        label_box_revision(&self.draft)
    }

    /// Active drawing class clamped to `0..num_classes`.
    pub fn clamped_draw_class(&self, num_classes: usize) -> u32 {
        if num_classes == 0 { 0 } else { self.draw_class.min((num_classes - 1) as u32) }
    }

    /// Synchronizes the editor with the currently displayed server `image`.
    ///
    /// - When switching to a different job or image, resets `base`/`draft` and history to `image.boxes`.
    /// - When the same image's server boxes change while clean, adopts the new server boxes and clears history.
    /// - When the same image's server boxes change while dirty (and not from our own in-flight save),
    ///   updates `base` to the latest server boxes and flags `changed_elsewhere` while preserving `draft`.
    pub fn sync_image(&mut self, job_id: &str, image: &LabelImage, num_classes: usize, saving: bool) {
        if num_classes > 0 {
            self.draw_class = self.draw_class.min((num_classes - 1) as u32);
        } else {
            self.draw_class = 0;
        }
        if !self.is_for(job_id, &image.id) {
            self.job_id = Some(job_id.to_string());
            self.image_id = Some(image.id.clone());
            self.base = image.boxes.clone();
            self.expect_rev = label_box_revision(&image.boxes);
            self.draft = image.boxes.clone();
            self.undo.clear();
            self.redo.clear();
            self.drawing = None;
            self.changed_elsewhere = false;
            self.pending_switch = None;
            return;
        }
        if same_boxes(&self.base, &image.boxes) {
            return;
        }
        if same_boxes(&self.draft, &self.base) {
            // Clean draft: adopt the fresh server boxes transparently.
            self.base = image.boxes.clone();
            self.expect_rev = label_box_revision(&image.boxes);
            self.draft = image.boxes.clone();
            self.undo.clear();
            self.redo.clear();
            self.changed_elsewhere = false;
        } else if !saving {
            // Dirty draft changed elsewhere (another tab, CLI, or agent): keep user's edits safe
            // and keep `expect_rev` at the revision this draft was edited against so a plain Ctrl+S
            // cannot silently overwrite the remote edit; update `base` so Discard adopts server boxes
            // and explicit Overwrite can send `base_revision()`.
            self.base = image.boxes.clone();
            self.changed_elsewhere = !same_boxes(&self.draft, &self.base);
            if !self.changed_elsewhere {
                self.expect_rev = label_box_revision(&self.base);
            }
        }
    }

    /// Applies a new `next` box vector to `draft`, pushing the previous `draft` onto the undo stack.
    fn push_edit(&mut self, next: Vec<BBox>) -> bool {
        if same_boxes(&self.draft, &next) {
            return false;
        }
        self.undo.push(std::mem::replace(&mut self.draft, next));
        if self.undo.len() > MAX_HISTORY {
            self.undo.remove(0);
        }
        self.redo.clear();
        self.pending_switch = None;
        if !self.is_dirty() {
            self.changed_elsewhere = false;
            self.expect_rev = label_box_revision(&self.base);
        }
        true
    }

    /// Appends a newly drawn box to `draft` and returns its index.
    pub fn add_box(&mut self, bbox: BBox) -> usize {
        let mut next = self.draft.clone();
        next.push(bbox);
        let idx = next.len() - 1;
        self.push_edit(next);
        idx
    }

    /// Updates `draft[index].class = class`. Returns `true` if the class actually changed.
    pub fn reclassify_box(&mut self, index: usize, class: u32) -> bool {
        let Some(existing) = self.draft.get(index) else {
            return false;
        };
        if existing.class == class {
            return false;
        }
        let mut next = self.draft.clone();
        next[index].class = class;
        self.push_edit(next)
    }

    /// Deletes `draft[index]`. Returns `true` if a box was removed.
    pub fn delete_box(&mut self, index: usize) -> bool {
        if index >= self.draft.len() {
            return false;
        }
        let mut next = self.draft.clone();
        next.remove(index);
        self.push_edit(next)
    }

    /// Steps back one edit (`Ctrl+Z`).
    pub fn undo(&mut self) -> bool {
        let Some(prev) = self.undo.pop() else {
            return false;
        };
        self.redo.push(std::mem::replace(&mut self.draft, prev));
        if self.redo.len() > MAX_HISTORY {
            self.redo.remove(0);
        }
        if !self.is_dirty() {
            self.changed_elsewhere = false;
            self.pending_switch = None;
            self.expect_rev = label_box_revision(&self.base);
        }
        true
    }

    /// Steps forward one edit (`Ctrl+Shift+Z` / `Ctrl+Y`).
    pub fn redo(&mut self) -> bool {
        let Some(next) = self.redo.pop() else {
            return false;
        };
        self.undo.push(std::mem::replace(&mut self.draft, next));
        if self.undo.len() > MAX_HISTORY {
            self.undo.remove(0);
        }
        if !self.is_dirty() {
            self.changed_elsewhere = false;
            self.pending_switch = None;
            self.expect_rev = label_box_revision(&self.base);
        }
        true
    }

    /// Reverts `draft` back to `base`.
    ///
    /// For normal local edits, pushes onto the undo stack so Discard itself is undoable (`Ctrl+Z`).
    /// When `changed_elsewhere` is true ("Load server boxes" after a concurrent remote edit), clears
    /// pre-conflict undo/redo snapshots so `Ctrl+Z` cannot restore stale boxes under the new server
    /// revision and bypass the conflict safeguard.
    pub fn discard(&mut self) -> bool {
        self.drawing = None;
        self.pending_switch = None;
        let was_conflict = self.changed_elsewhere;
        self.changed_elsewhere = false;
        self.expect_rev = label_box_revision(&self.base);
        let base = self.base.clone();
        if was_conflict {
            let changed = !same_boxes(&self.draft, &base);
            self.draft = base;
            self.undo.clear();
            self.redo.clear();
            changed
        } else {
            self.push_edit(base)
        }
    }

    /// Marks the current draft as saved after the server confirms `PUT /api/label/jobs/{id}/images/{img}`.
    /// Keeps the undo stack intact so the user can still undo past a save and re-save.
    pub fn commit_saved(&mut self, saved_boxes: Vec<BBox>) {
        self.expect_rev = label_box_revision(&saved_boxes);
        self.base = saved_boxes.clone();
        self.draft = saved_boxes;
        self.changed_elsewhere = false;
        self.pending_switch = None;
        self.drawing = None;
    }

    /// Begins a click-drag box gesture on the canvas.
    pub fn begin_draw(&mut self, start_img: (f32, f32), start_px: (f32, f32), class: u32) {
        self.drawing =
            Some(BoxDraft { start_img, cur_img: start_img, start_px, cur_px: start_px, class, moved: false });
    }

    /// Updates an active click-drag box gesture; returns `true` if state changed.
    pub fn update_draw(&mut self, cur_img: (f32, f32), cur_px: (f32, f32)) -> bool {
        let Some(d) = self.drawing.as_mut() else {
            return false;
        };
        d.cur_img = cur_img;
        d.cur_px = cur_px;
        let dx = cur_px.0 - d.start_px.0;
        let dy = cur_px.1 - d.start_px.1;
        if dx * dx + dy * dy >= DRAG_THRESHOLD_PX * DRAG_THRESHOLD_PX {
            d.moved = true;
        }
        true
    }

    /// Returns the live preview `BBox` while dragging, if moved past threshold and >= `MIN_BOX_PX`.
    pub fn preview_box(&self, img_w: f32, img_h: f32) -> Option<BBox> {
        let d = self.drawing.as_ref()?;
        if !d.moved {
            return None;
        }
        let (x1, y1, x2, y2) = clamp_box_coords(d.start_img, d.cur_img, img_w, img_h)?;
        Some(BBox { class: d.class, x1, y1, x2, y2 })
    }

    /// Finishes an active drag gesture. If the box meets minimum size, appends it to `draft` and
    /// returns `Some((new_index, true))`; otherwise returns `None` (no box created).
    pub fn finish_draw(&mut self, img_w: f32, img_h: f32) -> Option<usize> {
        let d = self.drawing.take()?;
        if !d.moved {
            return None;
        }
        let (x1, y1, x2, y2) = clamp_box_coords(d.start_img, d.cur_img, img_w, img_h)?;
        Some(self.add_box(BBox { class: d.class, x1, y1, x2, y2 }))
    }

    /// Cancels an active drag gesture without modifying `draft`.
    pub fn cancel_draw(&mut self) -> bool {
        self.drawing.take().is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{LabelImageStatus, LabelSource};

    fn sample_image(id: &str, boxes: Vec<BBox>) -> LabelImage {
        LabelImage {
            id: id.into(),
            file: format!("{id}.jpg"),
            width: 640,
            height: 480,
            status: LabelImageStatus::Labeled,
            boxes,
            source: Some(LabelSource::Agent),
            error: None,
            reviewed: false,
        }
    }

    #[test]
    fn clamp_box_coords_orders_clamps_rounds_and_rejects_tiny_or_non_finite() {
        // Normal drag bottom-right to top-left reverses cleanly and clamps to image bounds.
        let got = clamp_box_coords((700.0, 500.0), (-20.0, 10.24), 640.0, 480.0).unwrap();
        assert_eq!(got, (0.0, 10.2, 640.0, 480.0));

        // Tiny boxes under MIN_BOX_PX are rejected.
        assert_eq!(clamp_box_coords((10.0, 10.0), (13.0, 50.0), 640.0, 480.0), None);
        assert_eq!(clamp_box_coords((10.0, 10.0), (50.0, 13.5), 640.0, 480.0), None);
        assert!(clamp_box_coords((10.0, 10.0), (14.0, 14.0), 640.0, 480.0).is_some());

        // Non-finite inputs never produce NaN.
        assert_eq!(clamp_box_coords((f32::NAN, 10.0), (50.0, 50.0), 640.0, 480.0), None);
        assert_eq!(clamp_box_coords((10.0, 10.0), (50.0, f32::INFINITY), 640.0, 480.0), None);
        assert_eq!(clamp_box_coords((10.0, 10.0), (50.0, 50.0), 2.0, 480.0), None);
    }

    #[test]
    fn draw_relabel_delete_undo_redo_and_discard_state_machine() {
        let mut ed = LabelEditor::default();
        let b0 = BBox { class: 0, x1: 10.0, y1: 20.0, x2: 110.0, y2: 120.0 };
        let img = sample_image("li_1", vec![b0]);
        ed.sync_image("lb_1", &img, 3, false);

        assert!(!ed.is_dirty());
        assert!(!ed.can_undo());
        assert!(!ed.can_redo());

        // Click without dragging past threshold does not create a box.
        ed.begin_draw((50.0, 50.0), (100.0, 100.0), 1);
        ed.update_draw((51.0, 51.0), (101.0, 101.0));
        assert!(ed.preview_box(640.0, 480.0).is_none());
        assert_eq!(ed.finish_draw(640.0, 480.0), None);
        assert!(!ed.is_dirty());

        // Real drag creates a new box with live preview.
        ed.begin_draw((200.0, 100.0), (200.0, 100.0), 2);
        ed.update_draw((320.0, 240.0), (260.0, 170.0));
        let prev = ed.preview_box(640.0, 480.0).expect("live preview");
        assert_eq!(prev, BBox { class: 2, x1: 200.0, y1: 100.0, x2: 320.0, y2: 240.0 });
        let idx = ed.finish_draw(640.0, 480.0).expect("new box committed");
        assert_eq!(idx, 1);
        assert!(ed.is_dirty());
        assert_eq!(ed.draft.len(), 2);

        // Relabel new box from class 2 to class 1.
        assert!(ed.reclassify_box(1, 1));
        assert_eq!(ed.draft[1].class, 1);
        assert!(!ed.reclassify_box(1, 1), "no-op reclassify does not push undo");

        // Delete box 0.
        assert!(ed.delete_box(0));
        assert_eq!(ed.draft.len(), 1);
        assert_eq!(ed.draft[0].class, 1);

        // Undo delete -> restores box 0.
        assert!(ed.undo());
        assert_eq!(ed.draft.len(), 2);
        assert_eq!(ed.draft[1].class, 1);

        // Undo relabel -> box 1 back to class 2.
        assert!(ed.undo());
        assert_eq!(ed.draft[1].class, 2);

        // Undo draw -> back to base, no longer dirty!
        assert!(ed.undo());
        assert_eq!(ed.draft.len(), 1);
        assert!(!ed.is_dirty());
        assert!(!ed.can_undo());
        assert!(ed.can_redo());

        // Redo draw & relabel.
        assert!(ed.redo());
        assert_eq!(ed.draft.len(), 2);
        assert!(ed.is_dirty());
        assert!(ed.redo());
        assert_eq!(ed.draft[1].class, 1);

        // Discard reverts to base, and is itself undoable!
        assert!(ed.discard());
        assert!(!ed.is_dirty());
        assert_eq!(ed.draft.len(), 1);
        assert!(ed.undo(), "Discard can be undone");
        assert!(ed.is_dirty());
        assert_eq!(ed.draft.len(), 2);

        // Commit saved updates base and clears dirty while keeping undo history.
        let saved = ed.draft.clone();
        ed.commit_saved(saved);
        assert!(!ed.is_dirty());
        assert!(ed.can_undo());
    }

    #[test]
    fn remote_sse_update_preserves_dirty_draft_and_flags_conflict() {
        let mut ed = LabelEditor::default();
        let b0 = BBox { class: 0, x1: 10.0, y1: 10.0, x2: 100.0, y2: 100.0 };
        let b_remote = BBox { class: 2, x1: 30.0, y1: 30.0, x2: 150.0, y2: 150.0 };
        ed.sync_image("lb_1", &sample_image("li_1", vec![b0]), 3, false);

        // Clean draft adopts remote update transparently.
        ed.sync_image("lb_1", &sample_image("li_1", vec![b_remote]), 3, false);
        assert_eq!(ed.draft, vec![b_remote]);
        assert!(!ed.changed_elsewhere);

        // Dirty draft preserves local edits, updates base revision, and sets changed_elsewhere.
        ed.add_box(BBox { class: 1, x1: 200.0, y1: 200.0, x2: 300.0, y2: 300.0 });
        assert!(ed.is_dirty());
        let remote_v2 = vec![b0, b_remote];
        ed.sync_image("lb_1", &sample_image("li_1", remote_v2.clone()), 3, false);
        assert!(ed.changed_elsewhere);
        assert_eq!(ed.base, remote_v2);
        assert_eq!(ed.draft.len(), 2);
        assert_eq!(ed.draft[1].class, 1, "user draft box stays intact");
        assert_eq!(ed.base_revision(), label_box_revision(&remote_v2));
    }
}
