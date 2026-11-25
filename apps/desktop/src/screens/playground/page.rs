//! The Playground page's layout, and what it shows: the boxes that pass and the one inspected.

use folo_ui::{
    AnyElement, Context, IntoElement, ParentElement, StatefulInteractiveElement, Styled, Window, div, px,
};

use crate::api::{PredictResponse, ScoredBox};
use crate::app::{FoloApp, act};
use crate::controls::Btn;
use crate::screens::page_header;
use crate::theme::theme;
use crate::ui::{Tone, plural};
use crate::viz::DrawBox;

/// Filters `boxes` by confidence threshold and optional class index.
pub(crate) fn filter_detections(boxes: &[ScoredBox], conf: f32, class: Option<u32>) -> Vec<&ScoredBox> {
    boxes.iter().filter(|b| b.score >= conf && class.is_none_or(|c| b.class == c)).collect()
}

/// Formats the currently visible Playground detections as pretty-printed JSON for clipboard export.
pub(crate) fn detections_json(
    model: &str,
    image: &str,
    conf: f32,
    iou: f32,
    class_filter: Option<u32>,
    boxes: &[&ScoredBox],
) -> String {
    #[derive(serde::Serialize)]
    struct Export<'a> {
        model: &'a str,
        image: &'a str,
        conf: f32,
        iou: f32,
        #[serde(skip_serializing_if = "Option::is_none")]
        class_filter: Option<u32>,
        detections: &'a [&'a ScoredBox],
    }
    serde_json::to_string_pretty(&Export { model, image, conf, iou, class_filter, detections: boxes })
        .unwrap_or_else(|_| "{}".to_string())
}

impl FoloApp {
    /// Boxes of the current prediction that pass the confidence threshold and active class filter.
    pub(crate) fn visible_boxes(&self, p: &PredictResponse, active: Option<usize>) -> Vec<DrawBox> {
        filter_detections(&p.boxes, self.play_conf, self.play_class)
            .into_iter()
            .enumerate()
            .map(|(i, b)| DrawBox {
                label: Some(b.name.clone().into()),
                score: Some(b.score),
                highlight: active == Some(i),
                dim: active.is_some_and(|h| h != i),
                ..DrawBox::new(b.class, b.x1, b.y1, b.x2, b.y2)
            })
            .collect()
    }

    /// Copies the currently filtered Playground detections as formatted JSON and shows a toast.
    pub(crate) fn copy_play_json(&mut self, cx: &mut Context<Self>) {
        let Some(p) = self.play_pred.ready() else { return };
        let vis = filter_detections(&p.boxes, self.play_conf, self.play_class);
        let model =
            self.play_shown_model().map(|m| m.name.as_str()).or(self.play_model.as_deref()).unwrap_or("");
        let image = self.play_image.as_ref().map(|i| i.name.as_ref()).unwrap_or("");
        let json = detections_json(model, image, self.play_conf, self.play_iou, self.play_class, &vis);
        let body = plural(vis.len(), "detection");
        cx.write_to_clipboard(folo_ui::ClipboardItem::new_string(json));
        self.toast(Tone::Success, "Copied detection JSON", Some(body.into()), cx);
    }

    pub fn playground(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let no_models_hdr = self.store.models.ready().is_some_and(|m| m.is_empty());
        let open = if self.store.models.ready().is_some_and(|m| m.is_empty()) {
            self.button_disabled(
                "play-open",
                "Open image",
                Btn::Secondary,
                Some("icons/image-up.svg"),
                "Train a model first",
                window,
                cx,
            )
            .into_any_element()
        } else {
            self.button(
                "play-open",
                "Open image",
                Btn::Secondary,
                Some("icons/image-up.svg"),
                window,
                cx,
                act(|this, w, cx| this.pick_image(w, cx)),
            )
            .tooltip(crate::controls::tip("Open an image", Some("Ctrl O")))
            .into_any_element()
        };
        let header = page_header(
            &t,
            "Playground",
            if no_models_hdr {
                "Run a model on any image."
            } else {
                "Run a model on any image. Drop a file anywhere in the window."
            },
            // No models: the body's call to action is the only one (no inert button beside it).
            if no_models_hdr { Vec::new() } else { vec![open] },
        );
        let stage = self.stage(window, cx);
        self.probe_local_gpu(cx);
        // Without models the side panel has nothing to offer: the empty state gets the full width.
        let no_models = self.store.models.ready().is_some_and(|m| m.is_empty());
        let side = (!no_models).then(|| self.play_side(window, cx));
        div()
            .size_full()
            .flex()
            .flex_col()
            .gap(px(16.))
            .px(px(self.gutter()))
            .pt(px(20.))
            .pb(px(20.))
            .child(header)
            .child(div().flex_1().min_h_0().flex().gap(px(16.)).child(stage).children(side))
            .into_any_element()
    }

    /// The detection being inspected: under the pointer, or a keyboard-focused row.
    /// While the keyboard is in use the focused row leads (a pointer hover left on another row
    /// doesn't win); otherwise the hover does, as in the inspector.
    pub fn play_active(&self, window: &Window, cx: &mut Context<Self>) -> Option<usize> {
        let n = self
            .play_pred
            .ready()
            .map_or(0, |p| filter_detections(&p.boxes, self.play_conf, self.play_class).len());
        let focused = (0..n).find(|i| self.fh(format!("det-{i}"), cx).is_focused(window));
        if self.kb { focused.or(self.play_hover) } else { self.play_hover.or(focused) }
    }
}
