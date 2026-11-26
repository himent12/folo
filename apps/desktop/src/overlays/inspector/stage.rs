//! Interactive image stage of the inspector: canvas painting, wheel zoom, drag pan, hover hit-test.

use std::cell::Cell;
use std::sync::Arc;

use folo_ui::{
    AnyElement, Context, InteractiveElement, IntoElement, MouseButton, MouseDownEvent, MouseMoveEvent,
    ParentElement, RenderImage, ScrollWheelEvent, StatefulInteractiveElement, Styled, Window, canvas, div,
    px,
};

use crate::app::{FoloApp, act};
use crate::store::ImgKind;
use crate::theme::{Theme, radius, ty};
use crate::ui::*;
use crate::viz::{DrawBox, Fit, hit_box, paint_image_with_boxes, to_image, zoom_about};

thread_local! {
    /// Whether the active left-button press on the inspector stage engaged drag-panning past the
    /// click deadzone (cleared on press/click), so dragging out and back to the press point never
    /// triggers click-to-zoom on release.
    static STAGE_PANNED: Cell<bool> = const { Cell::new(false) };
}

/// Clears the active stage-pan flag (called on zoom reset, outside release, or wheel zoom).
pub(super) fn clear_stage_panned() {
    STAGE_PANNED.with(|p| p.set(false));
}

impl FoloApp {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn inspector_stage(
        &mut self,
        t: &Theme,
        ds_id: &str,
        img_id: &str,
        img: Option<Arc<RenderImage>>,
        dims: (f32, f32),
        boxes: &[DrawBox],
        zoom: f32,
        pan: (f32, f32),
        show_boxes: bool,
        show_labels: bool,
        window: &Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        // An image that would not load (its automatic retries may be spent) says so, with a Retry
        // that asks for it again (Tab reaches it; Enter or Space presses it). A full-size one shown
        // whose thumbnail failed after it loaded says so too: the grid would stay blank.
        let full_failed = self.image_failed(ds_id, img_id, ImgKind::Full);
        let thumb_failed = img.is_some() && self.image_failed(ds_id, img_id, ImgKind::Thumb);
        let unavailable = (full_failed || thumb_failed).then(|| {
            let (ds, id) = (ds_id.to_string(), img_id.to_string());
            let retry = self.button(
                "insp-retry-image",
                "Retry",
                crate::controls::Btn::Secondary,
                Some("icons/refresh-cw.svg"),
                window,
                cx,
                act(move |this, _, cx| this.retry_failed_image(&ds, &id, cx)),
            );
            div().absolute().bottom(px(40.)).left_0().right_0().flex().justify_center().child(
                div()
                    .flex()
                    .items_center()
                    .gap(px(12.))
                    .px(px(12.))
                    .py(px(8.))
                    .rounded(px(radius::MD))
                    .bg(t.bg_overlay)
                    .child(text(ty::SM, t.text_secondary).child(if !full_failed {
                        "Thumbnail unavailable: the grid can't show it"
                    } else if img.is_some() {
                        "Full-size image unavailable: showing the thumbnail"
                    } else {
                        "Image unavailable"
                    }))
                    .child(retry),
            )
        });
        let bounds_cell = self.insp_view.clone();
        // A pan pushed past the edge is stored clamped, so reversing it responds at once.
        if let (Some(b), Some(i)) = (bounds_cell.get(), self.inspector.as_mut()) {
            i.pan = crate::viz::clamp_pan(b, dims.0, dims.1, Fit::Contain, i.zoom, i.pan);
        }
        let paint_cell = bounds_cell.clone();
        let wheel_cell = bounds_cell.clone();
        let down_cell = bounds_cell.clone();
        let up_cell = bounds_cell.clone();
        let paint_boxes = if show_boxes { boxes.to_vec() } else { Vec::new() };
        let hit_boxes = paint_boxes.clone();
        let up_boxes = paint_boxes.clone();
        let hovering_stage_box = self.inspector.as_ref().is_some_and(|i| i.hover_from_stage);
        let tt = *t;
        let img2 = img;
        div()
            .id("insp-stage")
            .relative()
            .flex_1()
            .min_w_0()
            .bg(t.bg_inset)
            .cursor(if zoom > 1.0 {
                folo_ui::CursorStyle::OpenHand
            } else if hovering_stage_box {
                folo_ui::CursorStyle::PointingHand
            } else {
                folo_ui::CursorStyle::Arrow
            })
            .on_scroll_wheel(cx.listener(move |this, ev: &ScrollWheelEvent, w, cx| {
                let Some(b) = wheel_cell.get() else { return };
                let Some(insp) = this.inspector.as_mut() else { return };
                let dy = f32::from(ev.delta.pixel_delta(w.line_height()).y);
                let factor = (1.0 + dy * 0.0025).clamp(0.5, 2.0);
                let new_zoom = (insp.zoom * factor).clamp(1.0, 12.0);
                // Zoom about the cursor: keep the image point under it fixed.
                insp.pan = zoom_about(b, dims, insp.zoom, insp.pan, ev.position, new_zoom);
                insp.zoom = new_zoom;
                insp.drag_from = None;
                insp.focused_box = None;
                clear_stage_panned();
                w.prevent_default();
                cx.stop_propagation();
                cx.notify();
            }))
            .on_hover(cx.listener(|this, h: &bool, _, cx| {
                if !*h
                    && let Some(insp) = this.inspector.as_mut()
                    && insp.hover_from_stage
                {
                    insp.hover_from_stage = false;
                    if insp.hover_box.take().is_some() {
                        cx.notify();
                    }
                }
            }))
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(move |this, ev: &MouseDownEvent, _, cx| {
                    let _ = down_cell.get();
                    STAGE_PANNED.with(|p| p.set(false));
                    if let Some(insp) = this.inspector.as_mut() {
                        insp.drag_from = Some((ev.position, insp.pan));
                        cx.notify();
                    }
                }),
            )
            .on_mouse_up(
                MouseButton::Left,
                cx.listener(|this, _, _, cx| {
                    if let Some(insp) = this.inspector.as_mut() {
                        insp.drag_from = None;
                        cx.notify();
                    }
                }),
            )
            .on_mouse_up_out(
                MouseButton::Left,
                cx.listener(|this, _, _, cx| {
                    clear_stage_panned();
                    if let Some(insp) = this.inspector.as_mut()
                        && insp.drag_from.take().is_some()
                    {
                        cx.notify();
                    }
                }),
            )
            .on_click(cx.listener(move |this, ev: &folo_ui::ClickEvent, _, cx| {
                let folo_ui::ClickEvent::Mouse(m) = ev else { return };
                let panned = STAGE_PANNED.with(|p| p.replace(false));
                let dx = f32::from(m.up.position.x - m.down.position.x);
                let dy = f32::from(m.up.position.y - m.down.position.y);
                if panned || dx * dx + dy * dy > 16.0 {
                    return;
                }
                let hit = this.inspector.as_ref().and_then(|insp| {
                    if !insp.boxes {
                        return None;
                    }
                    let b = up_cell.get()?;
                    if !b.contains(&m.up.position) {
                        return None;
                    }
                    let (x, y) = to_image(b, dims, Fit::Contain, insp.zoom, insp.pan, m.up.position);
                    let idx = hit_box(&up_boxes, x, y)?;
                    let bx = &up_boxes[idx];
                    Some((idx, (bx.x1, bx.y1, bx.x2, bx.y2), b))
                });
                if let Some((idx, coords, b)) = hit {
                    this.focus_inspector_box(idx, dims, coords, cx);
                    // Re-sync stage hover against the updated zoom/pan so returning to Fit never
                    // leaves stale highlight or `hover_from_stage = false` under the pointer.
                    if let Some(insp) = this.inspector.as_mut() {
                        let (x, y) = to_image(b, dims, Fit::Contain, insp.zoom, insp.pan, m.up.position);
                        let next_hit = if insp.boxes { hit_box(&up_boxes, x, y) } else { None };
                        insp.hover_box = next_hit;
                        insp.hover_from_stage = next_hit.is_some();
                    }
                }
            }))
            .on_mouse_move(cx.listener(move |this, ev: &MouseMoveEvent, _, cx| {
                let Some(insp) = this.inspector.as_mut() else { return };
                if insp.drag_from.is_some() && ev.pressed_button != Some(MouseButton::Left) {
                    insp.drag_from = None;
                    cx.notify();
                }
                if let Some((from, pan0)) = insp.drag_from {
                    if insp.zoom > 1.0 {
                        let dx = f32::from(ev.position.x - from.x);
                        let dy = f32::from(ev.position.y - from.y);
                        if STAGE_PANNED.with(Cell::get)
                            || dx * dx + dy * dy > 16.0
                            || (insp.pan.0 - pan0.0).abs() > 0.5
                            || (insp.pan.1 - pan0.1).abs() > 0.5
                        {
                            STAGE_PANNED.with(|p| p.set(true));
                            insp.pan = (pan0.0 + dx, pan0.1 + dy);
                            insp.focused_box = None;
                            cx.notify();
                        }
                    }
                    return;
                }
                if !insp.boxes {
                    if insp.hover_from_stage && insp.hover_box.take().is_some() {
                        insp.hover_from_stage = false;
                        cx.notify();
                    }
                    return;
                }
                let Some(b) = bounds_cell.get() else { return };
                let hit = if b.contains(&ev.position) {
                    let (x, y) = to_image(b, dims, Fit::Contain, insp.zoom, insp.pan, ev.position);
                    hit_box(&hit_boxes, x, y)
                } else {
                    None
                };
                if hit != insp.hover_box || insp.hover_from_stage != hit.is_some() {
                    insp.hover_box = hit;
                    insp.hover_from_stage = hit.is_some();
                    // The pointer moved over the image: hover leads again (not a keyboard
                    // focus left on a label row).
                    this.kb = false;
                    cx.notify();
                }
            }))
            .child(
                canvas(
                    |_, _, _| (),
                    move |b, _, window, cx| {
                        let pad_x = (b.size.width / 2.).min(px(24.));
                        let pad_y = (b.size.height / 2.).min(px(24.));
                        let inner = folo_ui::Bounds::new(
                            b.origin + folo_ui::point(pad_x, pad_y),
                            // 24 px on every side when there is room (the zoom controls live in the header now).
                            b.size - folo_ui::size(pad_x * 2., pad_y * 2.),
                        );
                        paint_cell.set(Some(inner));
                        window.with_content_mask(Some(folo_ui::ContentMask { bounds: b }), |window| {
                            paint_image_with_boxes(
                                &tt,
                                window,
                                cx,
                                inner,
                                img2.as_ref(),
                                dims,
                                &paint_boxes,
                                Fit::Contain,
                                zoom,
                                pan,
                                show_labels,
                                false,
                                0.,
                            );
                        });
                    },
                )
                .size_full(),
            )
            .children(unavailable)
            .into_any_element()
    }
}
