//! The Playground page's layout, and what it shows: the boxes that pass and the one inspected.

use gpui::{
    AnyElement, Context, IntoElement, ParentElement, StatefulInteractiveElement, Styled, Window, div, px,
};

use crate::api::PredictResponse;
use crate::app::{FoloApp, act};
use crate::controls::Btn;
use crate::screens::page_header;
use crate::theme::theme;
use crate::viz::DrawBox;

impl FoloApp {
    /// Boxes of the current prediction that pass the confidence threshold.
    pub(super) fn visible_boxes(&self, p: &PredictResponse, active: Option<usize>) -> Vec<DrawBox> {
        p.boxes
            .iter()
            .filter(|b| b.score >= self.play_conf)
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
                act(|this, _, cx| this.pick_image(cx)),
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
            .map_or(0, |p| p.boxes.iter().filter(|b| b.score >= self.play_conf).count());
        let focused = (0..n).find(|i| self.fh(format!("det-{i}"), cx).is_focused(window));
        if self.kb { focused.or(self.play_hover) } else { self.play_hover.or(focused) }
    }
}
