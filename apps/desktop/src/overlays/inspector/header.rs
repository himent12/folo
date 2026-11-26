//! Top header bar of the image inspector: stepping, box/label toggles, Playground jump, zoom, close.

use folo_ui::{
    AnyElement, Context, IntoElement, ParentElement, StatefulInteractiveElement, Styled, Window, div, px,
};

use crate::app::{FoloApp, Route, act};
use crate::controls::Btn;
use crate::theme::{Theme, ty};
use crate::ui::*;

impl FoloApp {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn inspector_header(
        &mut self,
        t: &Theme,
        ds_id: &str,
        img_id: &str,
        sample_run: Option<&str>,
        counter: String,
        zoom: f32,
        show_boxes: bool,
        show_labels: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let prev = self.icon_button(
            "insp-prev",
            "icons/chevron-left.svg",
            "Previous (←)",
            window,
            cx,
            act(|this, _, cx| this.step_inspector(-1, cx)),
        );
        let next = self.icon_button(
            "insp-next",
            "icons/chevron-right.svg",
            "Next (→)",
            window,
            cx,
            act(|this, _, cx| this.step_inspector(1, cx)),
        );
        let close = self.icon_button(
            "insp-close",
            "icons/x.svg",
            "Close (Esc)",
            window,
            cx,
            act(|this, w, cx| this.close_inspector(w, cx)),
        );
        let boxes_toggle = self
            .chip(
                "insp-boxes-toggle",
                "Boxes",
                show_boxes,
                None,
                None,
                window,
                cx,
                act(|this, _, cx| this.toggle_inspector_boxes(cx)),
            )
            .tooltip(crate::controls::tip("Toggle bounding boxes", Some("B")));
        let labels_tip = if show_boxes { "Toggle label chips" } else { "Show boxes and label chips" };
        let labels_toggle = self
            .chip(
                "insp-labels-toggle",
                "Labels",
                show_boxes && show_labels,
                None,
                None,
                window,
                cx,
                act(|this, _, cx| this.toggle_inspector_labels(cx)),
            )
            .tooltip(crate::controls::tip(labels_tip, Some("L")));
        let narrow = window.viewport_size().width < px(1060.);
        let pg_label = if narrow { "Playground" } else { "Try in Playground" };
        let playground_btn = match self.open_image_blocked() {
            Some(why) => self.button_disabled(
                "insp-playground",
                pg_label,
                Btn::Secondary,
                Some("icons/scan-eye.svg"),
                why,
                window,
                cx,
            ),
            None => {
                let pg_ds = ds_id.to_string();
                let pg_img = img_id.to_string();
                let sample_model =
                    sample_run.and_then(|run| self.store.run(run)).and_then(|r| r.model_id.clone());
                self.button(
                    "insp-playground",
                    pg_label,
                    Btn::Secondary,
                    Some("icons/scan-eye.svg"),
                    window,
                    cx,
                    act(move |this, w, cx| {
                        if let Some(mid) = sample_model.as_deref()
                            && this.store.model(mid).is_some()
                        {
                            this.pick_play_model(mid.to_string(), cx);
                        }
                        this.close_inspector(w, cx);
                        this.navigate(Route::Playground, w, cx);
                        this.play_dataset_image(&pg_ds, &pg_img, cx);
                    }),
                )
                .tooltip(crate::controls::tip("Run a model on this image in Playground", None))
            }
        };
        // Fitted by any path (the readout, 0, −, another image), focus left on the readout moves
        // to its neighbour.
        if zoom <= 1.0 && self.fh("insp-fit", cx).is_focused(window) {
            self.focus_key("insp-zoom-in", window, cx);
        }
        // Zoom in the top bar, right-aligned before Close, as Playground keeps it in its header.
        let zoom_ctrls = div()
            .flex()
            .items_center()
            .gap(px(2.))
            .child(self.icon_button(
                "insp-zoom-out",
                "icons/zoom-out.svg",
                "Zoom out",
                window,
                cx,
                act(|this, _, cx| this.zoom_by(1.0 / 1.4, cx)),
            ))
            .child(self.zoom_readout(
                "insp-fit",
                zoom,
                window,
                cx,
                act(|this, _, cx| {
                    let mut was_dragging = false;
                    if let Some(i) = this.inspector.as_mut() {
                        i.zoom = 1.0;
                        i.pan = (0., 0.);
                        was_dragging = i.drag_from.take().is_some();
                        i.focused_box = None;
                    }
                    super::stage::interrupt_stage_drag(was_dragging);
                    cx.notify();
                }),
            ))
            .child(self.icon_button(
                "insp-zoom-in",
                "icons/zoom-in.svg",
                "Zoom in",
                window,
                cx,
                act(|this, _, cx| this.zoom_by(1.4, cx)),
            ));
        div()
            .h(px(48.))
            .flex_none()
            .px(px(12.))
            .flex()
            .items_center()
            .gap(px(8.))
            .bg(t.bg_overlay)
            .border_b_1()
            .border_color(t.line_subtle)
            .child(prev)
            .child(next)
            .child(text(ty::SM, t.text_secondary).tnum().whitespace_nowrap().child(counter))
            .child(div().flex_1())
            .child(boxes_toggle)
            .child(labels_toggle)
            .child(div().w(px(1.)).h(px(16.)).bg(t.line_subtle))
            .child(playground_btn)
            .child(div().w(px(1.)).h(px(16.)).bg(t.line_subtle))
            .child(zoom_ctrls)
            .child(div().w(px(1.)).h(px(16.)).bg(t.line_subtle))
            .child(close)
            .into_any_element()
    }
}
