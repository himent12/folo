//! The Playground page's layout, and what it shows: the boxes that pass and the one inspected.

use folo_ui::{
    AnyElement, Context, IntoElement, ParentElement, StatefulInteractiveElement, Styled, Window, div, px,
};

use crate::api::{PredictResponse, ScoredBox};
use crate::app::{FoloApp, act};
use crate::controls::Btn;
use crate::screens::page_header;
use crate::theme::theme;
use crate::ui::Tone;
use crate::viz::DrawBox;

/// Filters `boxes` by confidence threshold and optional class index.
pub(crate) fn filter_detections(boxes: &[ScoredBox], conf: f32, class: Option<u32>) -> Vec<&ScoredBox> {
    boxes.iter().filter(|b| b.score >= conf && class.is_none_or(|c| b.class == c)).collect()
}

/// Formats the currently visible Playground detections as pretty-printed JSON for clipboard or file export.
pub(crate) fn detections_json(
    model: &str,
    image: &str,
    conf: f32,
    iou: Option<f32>,
    class_filter: Option<u32>,
    boxes: &[&ScoredBox],
) -> anyhow::Result<String> {
    #[derive(serde::Serialize)]
    struct Export<'a> {
        model: &'a str,
        image: &'a str,
        conf: f32,
        #[serde(skip_serializing_if = "Option::is_none")]
        iou: Option<f32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        class_filter: Option<u32>,
        detections: &'a [&'a ScoredBox],
    }
    Ok(serde_json::to_string_pretty(&Export { model, image, conf, iou, class_filter, detections: boxes })?)
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

    /// Boxes of Model B's prediction that pass the confidence threshold and active class filter.
    pub(crate) fn visible_boxes_b(&self, p: &PredictResponse, active: Option<usize>) -> Vec<DrawBox> {
        self.filter_detections_b(&p.boxes)
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

    /// Copies the currently filtered Playground detections (Single or Compare A/B) as formatted JSON.
    pub(crate) fn copy_play_json(&mut self, cx: &mut Context<Self>) {
        let Some(built) = self.build_play_export_json() else { return };
        match built {
            Ok((json, _, summary)) => {
                cx.write_to_clipboard(folo_ui::ClipboardItem::new_string(json));
                self.toast(Tone::Success, "Copied detection JSON", Some(summary.into()), cx);
            }
            Err(e) => {
                self.toast(Tone::Danger, "JSON export failed", Some(e.to_string().into()), cx);
            }
        }
    }

    pub fn playground(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let no_models_hdr = self.store.models.ready().is_some_and(|m| m.is_empty());
        let compare_toggle = self
            .segmented(
                "play-mode",
                &["Single", "Compare A/B"],
                usize::from(self.play_compare.enabled),
                window,
                cx,
                std::rc::Rc::new(|this, i, _, cx| {
                    this.set_play_compare(i == 1, cx);
                }),
            )
            .into_any_element();
        let open = if let Some(why) = self.open_image_blocked() {
            self.button_disabled(
                "play-open",
                "Open image",
                Btn::Secondary,
                Some("icons/image-up.svg"),
                why,
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
            } else if self.play_compare.enabled {
                "Compare two models side-by-side on any image. Drop a file anywhere in the window."
            } else {
                "Run a model on any image. Drop a file anywhere in the window."
            },
            // No models: the body's call to action is the only one (no inert button beside it).
            if no_models_hdr { Vec::new() } else { vec![compare_toggle, open] },
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
        let hover = self.play_hover.filter(|&i| i < n);
        if self.kb { focused.or(hover) } else { hover.or(focused) }
    }

    /// The Model B detection being inspected in Compare mode.
    pub fn play_active_b(&self, window: &Window, cx: &mut Context<Self>) -> Option<usize> {
        let n = self.play_compare.pred_b.ready().map_or(0, |p| self.filter_detections_b(&p.boxes).len());
        let focused = (0..n).find(|i| self.fh(format!("det-b-{i}"), cx).is_focused(window));
        let hover = self.play_compare.hover_b.filter(|&i| i < n);
        if self.kb { focused.or(hover) } else { hover.or(focused) }
    }
}
