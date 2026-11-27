//! Interactive bounding-box canvas stage for Agent Labeling: renders working draft boxes,
//! 8 resize handles on the selected box, and live click-drag preview; supports click-to-select,
//! drag-to-move, drag-handle-to-resize, click-drag-to-draw, and wheel zoom.

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, BorderStyle, Bounds, Context, CursorStyle, InteractiveElement, IntoElement, MouseButton,
    MouseDownEvent, MouseMoveEvent, ParentElement, ScrollWheelEvent, StatefulInteractiveElement, Styled,
    canvas, div, point, px, quad, size,
};

use crate::api::{ApiError, BBox, LabelImage, LabelImageStatus, LabelJob};
use crate::app::FoloApp;
use crate::label::{EditTool, SelectedBox, TransformKind, handle_points, hit_resize_handle};
use crate::store::ImgKind;
use crate::theme::{Theme, radius, tokens};
use crate::viz::{DrawBox, Fit, hit_box, image_rect, paint_image_with_boxes, to_image, zoom_about};

/// Hit radius in logical screen pixels for grabbing a corner/edge resize handle.
const HANDLE_HIT_RADIUS_PX: f32 = 8.0;

impl FoloApp {
    /// Interactive image stage displaying `draft_boxes`, resize handles on the selected box,
    /// and any active live click-drag create/move/resize preview.
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
        let transform_preview = self.label.editor.preview_transform(dims.0, dims.1);
        let is_transforming = self.label.editor.transforming.is_some();
        let hover = if is_drawing || is_transforming { None } else { self.label.hover_box.or(sel_box) };
        let mut draw: Vec<DrawBox> = if self.label.show_boxes {
            draft_boxes
                .iter()
                .enumerate()
                .map(|(i, orig)| {
                    let (b, live_transform) = match transform_preview {
                        Some((ti, tb)) if ti == i => (tb, true),
                        _ => (*orig, false),
                    };
                    let dim_by_filter = class_filter.is_some_and(|c| c != b.class);
                    let dim_by_hover = hover.is_some_and(|h| h != i);
                    let cls_name = job
                        .classes
                        .get(b.class as usize)
                        .cloned()
                        .unwrap_or_else(|| format!("class {}", b.class));
                    let label = if live_transform {
                        format!("{cls_name} {:.0}×{:.0}", b.x2 - b.x1, b.y2 - b.y1)
                    } else {
                        cls_name
                    };
                    DrawBox {
                        label: Some(label.into()),
                        highlight: live_transform
                            || (!is_drawing && (sel_box == Some(i) || self.label.hover_box == Some(i))),
                        dim: is_drawing
                            || (transform_preview.is_some() && !live_transform)
                            || dim_by_filter
                            || dim_by_hover,
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

        let locked = active.status == LabelImageStatus::Labeling;
        let show_handles = self.label.show_boxes
            && !locked
            && !self.label.mutating
            && !is_drawing
            && self.label.editor.tool == EditTool::Select;
        let handle_box: Option<BBox> = if show_handles {
            if let Some((_, tb)) = transform_preview {
                Some(tb)
            } else {
                sel_box.and_then(|i| draft_boxes.get(i).copied())
            }
        } else {
            None
        };

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
                if let Some(hb) = handle_box {
                    let rect = image_rect(b, dims.0, dims.1, Fit::Contain, zoom, pan);
                    let clip = b.intersect(&rect);
                    if clip.size.width > px(0.) && clip.size.height > px(0.) {
                        let halo: folo_ui::Hsla = folo_ui::rgba(tokens::CLASS_HALO).into();
                        let fill_c = t_copy.accent_solid;
                        let half = px(3.5);
                        let sz = size(px(7.), px(7.));
                        for (_, pt) in handle_points(hb, rect, dims) {
                            let r = Bounds::new(point(pt.x - half, pt.y - half), sz);
                            window.paint_quad(quad(r, px(1.), fill_c, px(1.), halo, BorderStyle::Solid));
                        }
                    }
                }
            },
        )
        .size_full();

        let active_id_click = active.id.clone();
        let draft_rev_click = draft_rev.to_string();
        let n_classes = job.classes.len();
        let sel_box_for_click = sel_box.and_then(|i| draft_boxes.get(i).copied().map(|b| (i, b)));
        let sel_box_for_hover = sel_box_for_click;

        let cursor = if let Some(tr) = &self.label.editor.transforming {
            match tr.kind {
                TransformKind::Move => CursorStyle::ClosedHand,
                TransformKind::Resize(h) => h.cursor(),
            }
        } else if is_drawing || self.label.editor.tool == EditTool::Draw {
            CursorStyle::Crosshair
        } else if let Some(h) = self.label.editor.hover_handle {
            h.cursor()
        } else if self.label.editor.hover_move {
            CursorStyle::OpenHand
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
            .border_color(if is_drawing || is_transforming { t.accent_line } else { t.line_subtle })
            .overflow_hidden()
            .cursor(cursor)
            .on_scroll_wheel(cx.listener(move |this, ev: &ScrollWheelEvent, w, cx| {
                let Some(b) = view_for_wheel.get() else { return };
                let dy = f32::from(ev.delta.pixel_delta(w.line_height()).y);
                // Ignore non-finite or zero deltas, and let downward scroll at 1x fit zoom bubble to the page.
                if !dy.is_finite() || dy == 0.0 || (dy < 0.0 && this.label.zoom <= 1.0 + 1e-4) {
                    return;
                }
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
                    if this.label.editor.tool == EditTool::Select {
                        // 1. Resize handle on currently selected box takes priority.
                        if let Some((si, sb)) = sel_box_for_click
                            && !locked
                            && !this.label.mutating
                        {
                            let rect =
                                image_rect(b, dims.0, dims.1, Fit::Contain, this.label.zoom, this.label.pan);
                            if let Some(handle) =
                                hit_resize_handle(sb, rect, dims, ev.position, HANDLE_HIT_RADIUS_PX)
                            {
                                this.label.editor.begin_transform(
                                    si,
                                    (ix, iy),
                                    px_pos,
                                    TransformKind::Resize(handle),
                                );
                                cx.notify();
                                return;
                            }
                        }
                        // 2. Clicking any box selects it and primes a Move drag (only commits past drag threshold).
                        if let Some(index) = hit_box(&draw_for_hit, ix, iy) {
                            this.label.selected_box = Some(SelectedBox {
                                image_id: active_id_click.clone(),
                                index,
                                revision: draft_rev_click.clone(),
                            });
                            if !locked && !this.label.mutating {
                                this.label.editor.begin_transform(
                                    index,
                                    (ix, iy),
                                    px_pos,
                                    TransformKind::Move,
                                );
                            }
                            cx.notify();
                            return;
                        }
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
                if this.label.editor.drawing.is_some()
                    || this.label.editor.transforming.is_some()
                    || this.label.editor.tool == EditTool::Draw
                {
                    return;
                }
                let Some(b) = view_for_hover.get() else { return };
                let mut changed = false;
                let (next_handle, next_move, hit) = if b.contains(&ev.position) {
                    let rect = image_rect(b, dims.0, dims.1, Fit::Contain, this.label.zoom, this.label.pan);
                    let handle = sel_box_for_hover.and_then(|(_, sb)| {
                        (!locked && !this.label.mutating)
                            .then(|| hit_resize_handle(sb, rect, dims, ev.position, HANDLE_HIT_RADIUS_PX))
                            .flatten()
                    });
                    let (ix, iy) =
                        to_image(b, dims, Fit::Contain, this.label.zoom, this.label.pan, ev.position);
                    let hit = hit_box(&draw_for_hover, ix, iy);
                    let over_sel = !locked
                        && !this.label.mutating
                        && handle.is_none()
                        && sel_box_for_hover.is_some_and(|(si, _)| hit == Some(si));
                    (handle, over_sel, hit)
                } else {
                    (None, false, None)
                };
                if next_handle != this.label.editor.hover_handle {
                    this.label.editor.hover_handle = next_handle;
                    changed = true;
                }
                if next_move != this.label.editor.hover_move {
                    this.label.editor.hover_move = next_move;
                    changed = true;
                }
                if hit != this.label.hover_box {
                    this.label.hover_box = hit;
                    changed = true;
                }
                if changed {
                    cx.notify();
                }
            }))
            .on_hover(cx.listener(|this, h: &bool, _, cx| {
                if !*h {
                    let mut changed = this.label.hover_box.take().is_some();
                    if this.label.editor.hover_handle.take().is_some() {
                        changed = true;
                    }
                    if std::mem::take(&mut this.label.editor.hover_move) {
                        changed = true;
                    }
                    if changed {
                        cx.notify();
                    }
                }
            }))
            .child(canvas_el)
            .when(failed, |d| d.child(crate::screens::unavailable_mark(t, true)))
            .into_any_element()
    }
}
