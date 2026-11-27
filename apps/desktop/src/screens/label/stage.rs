//! Interactive bounding-box canvas stage for Agent Labeling: renders working draft boxes and
//! live click-drag preview, supports click-to-select, click-drag-to-draw, and wheel zoom.

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, CursorStyle, InteractiveElement, IntoElement, MouseButton, MouseDownEvent,
    MouseMoveEvent, ParentElement, ScrollWheelEvent, StatefulInteractiveElement, Styled, canvas, div, px,
};

use crate::api::{ApiError, BBox, LabelImage, LabelImageStatus, LabelJob};
use crate::app::FoloApp;
use crate::label::{EditTool, SelectedBox};
use crate::store::ImgKind;
use crate::theme::{Theme, radius};
use crate::viz::{DrawBox, Fit, hit_box, paint_image_with_boxes, to_image, zoom_about};

impl FoloApp {
    /// Interactive image stage displaying `draft_boxes` plus any active live click-drag preview.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn label_stage(
        &mut self,
        t: &Theme,
        job: &LabelJob,
        active: &LabelImage,
        draft_boxes: &[BBox],
        draft_rev: &str,
        sel_box: Option<usize>,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let img = self
            .label_image_px(&job.id, &active.id, ImgKind::Full, cx)
            .or_else(|| self.label_image_px(&job.id, &active.id, ImgKind::Thumb, cx));
        let failed = img.is_none() && self.label_image_failed(&job.id, &active.id, ImgKind::Full);
        let dims = (active.width.max(1) as f32, active.height.max(1) as f32);
        let class_filter = self.label.class_filter;
        let is_drawing = self.label.editor.drawing.is_some();
        let hover = if is_drawing { None } else { self.label.hover_box.or(sel_box) };
        let mut draw: Vec<DrawBox> = if self.label.show_boxes {
            draft_boxes
                .iter()
                .enumerate()
                .map(|(i, b)| {
                    let dim_by_filter = class_filter.is_some_and(|c| c != b.class);
                    let dim_by_hover = hover.is_some_and(|h| h != i);
                    DrawBox {
                        label: job.classes.get(b.class as usize).cloned().map(Into::into),
                        highlight: !is_drawing && (sel_box == Some(i) || self.label.hover_box == Some(i)),
                        dim: is_drawing || dim_by_filter || dim_by_hover,
                        ..DrawBox::new(b.class, b.x1, b.y1, b.x2, b.y2)
                    }
                })
                .collect()
        } else {
            Vec::new()
        };
        let draw_for_hit = draw.clone();
        let draw_for_hover = draw.clone();

        // Append the live click-drag preview box on top of existing boxes.
        if let Some(prev) = self.label.editor.preview_box(dims.0, dims.1) {
            let cls_name = job
                .classes
                .get(prev.class as usize)
                .cloned()
                .unwrap_or_else(|| format!("class {}", prev.class));
            let label = format!("{cls_name} {:.0}×{:.0}", prev.x2 - prev.x1, prev.y2 - prev.y1);
            draw.push(DrawBox {
                label: Some(label.into()),
                highlight: true,
                ..DrawBox::new(prev.class, prev.x1, prev.y1, prev.x2, prev.y2)
            });
        }

        let t_copy = *t;
        let zoom = self.label.zoom;
        let pan = self.label.pan;
        let show_labels = self.label.show_labels;
        let view_cell = self.label.stage_view.clone();
        let view_for_click = self.label.stage_view.clone();
        let view_for_hover = self.label.stage_view.clone();
        let view_for_wheel = self.label.stage_view.clone();
        let canvas_el = canvas(
            move |b, _, _| {
                view_cell.set(Some(b));
            },
            move |b, _, window, cx| {
                paint_image_with_boxes(
                    &t_copy,
                    window,
                    cx,
                    b,
                    img.as_ref(),
                    dims,
                    &draw,
                    Fit::Contain,
                    zoom,
                    pan,
                    show_labels,
                    true,
                    0.,
                );
            },
        )
        .size_full();

        let locked = active.status == LabelImageStatus::Labeling;
        let active_id_click = active.id.clone();
        let draft_rev_click = draft_rev.to_string();
        let n_classes = job.classes.len();
        let cursor = if is_drawing || self.label.editor.tool == EditTool::Draw {
            CursorStyle::Crosshair
        } else {
            CursorStyle::PointingHand
        };

        div()
            .id("label-stage")
            .relative()
            .size_full()
            .rounded(px(radius::MD))
            .bg(t.bg_app)
            .border_1()
            .border_color(if is_drawing { t.accent_line } else { t.line_subtle })
            .overflow_hidden()
            .cursor(cursor)
            .on_scroll_wheel(cx.listener(move |this, ev: &ScrollWheelEvent, w, cx| {
                let Some(b) = view_for_wheel.get() else { return };
                let dy = f32::from(ev.delta.pixel_delta(w.line_height()).y);
                let new_zoom = (this.label.zoom * (1.0 + dy * 0.0025).clamp(0.5, 2.0)).clamp(1.0, 12.0);
                this.label.pan = zoom_about(b, dims, this.label.zoom, this.label.pan, ev.position, new_zoom);
                this.label.zoom = new_zoom;
                w.prevent_default();
                cx.stop_propagation();
                cx.notify();
            }))
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(move |this, ev: &MouseDownEvent, _, cx| {
                    let Some(b) = view_for_click.get() else { return };
                    let (ix, iy) =
                        to_image(b, dims, Fit::Contain, this.label.zoom, this.label.pan, ev.position);
                    let px_pos = (f32::from(ev.position.x), f32::from(ev.position.y));
                    if this.label.editor.tool == EditTool::Select
                        && let Some(index) = hit_box(&draw_for_hit, ix, iy)
                    {
                        this.label.selected_box = Some(SelectedBox {
                            image_id: active_id_click.clone(),
                            index,
                            revision: draft_rev_click.clone(),
                        });
                        cx.notify();
                        return;
                    }
                    if locked {
                        if this.label.editor.tool == EditTool::Select
                            && this.label.selected_box.take().is_some()
                        {
                            cx.notify();
                            return;
                        }
                        this.label.review_error = Some(ApiError::new(
                            "label_job_active",
                            "The agent is labeling this image right now.",
                        ));
                        cx.notify();
                        return;
                    }
                    if this.label.mutating || n_classes == 0 {
                        return;
                    }
                    let cls = this.label.editor.clamped_draw_class(n_classes);
                    this.label.editor.begin_draw((ix, iy), px_pos, cls);
                    cx.notify();
                }),
            )
            .on_mouse_move(cx.listener(move |this, ev: &MouseMoveEvent, _, cx| {
                if this.label.editor.drawing.is_some() || this.label.editor.tool == EditTool::Draw {
                    return;
                }
                let Some(b) = view_for_hover.get() else { return };
                let hit = if b.contains(&ev.position) {
                    let (ix, iy) =
                        to_image(b, dims, Fit::Contain, this.label.zoom, this.label.pan, ev.position);
                    hit_box(&draw_for_hover, ix, iy)
                } else {
                    None
                };
                if hit != this.label.hover_box {
                    this.label.hover_box = hit;
                    cx.notify();
                }
            }))
            .on_hover(cx.listener(|this, h: &bool, _, cx| {
                if !*h && this.label.hover_box.take().is_some() {
                    cx.notify();
                }
            }))
            .child(canvas_el)
            .when(failed, |d| d.child(crate::screens::unavailable_mark(t, true)))
            .into_any_element()
    }
}
