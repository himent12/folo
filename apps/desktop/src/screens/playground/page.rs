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
    /// Active effective class filter for Model A detections across the stage overlay, detection
    /// rail, hover indexing, Compare A/B, and exports. When batch mode is active with
    /// `BatchQaFilter::ByClass(cid)`, that class filter applies uniformly everywhere.
    pub(crate) fn play_effective_class(&self) -> Option<u32> {
        if self.play_batch.is_active() {
            self.play_batch.effective_class(self.play_class)
        } else {
            self.play_class
        }
    }

    /// True when batch mode has a non-`All` QA filter active that currently matches zero images.
    pub(crate) fn play_batch_qa_zero_match(&self) -> bool {
        self.play_batch.is_active()
            && self.play_batch.qa_filter != super::batch_qa::BatchQaFilter::All
            && self.play_batch_filtered_indices().is_empty()
    }

    /// Resolves the display name of Model A class `cid` from Model A's class list, the current
    /// prediction, or any fresh batch result (used for Compare A/B cross-model name matching).
    pub(crate) fn play_class_name(&self, cid: u32) -> Option<String> {
        self.play_shown_model()
            .and_then(|m| m.classes.get(cid as usize))
            .filter(|s| !s.trim().is_empty())
            .cloned()
            .or_else(|| {
                self.play_pred
                    .ready()
                    .and_then(|p| p.boxes.iter().find(|b| b.class == cid && !b.name.trim().is_empty()))
                    .map(|b| b.name.clone())
            })
            .or_else(|| {
                self.play_batch.items.iter().find_map(|item| match &item.status {
                    super::batch::BatchItemStatus::Done(res) => res
                        .pred
                        .boxes
                        .iter()
                        .find(|b| b.class == cid && !b.name.trim().is_empty())
                        .map(|b| b.name.clone()),
                    _ => None,
                })
            })
    }

    /// Boxes of the current prediction that pass the confidence threshold and active class filter.
    pub(crate) fn visible_boxes(&self, p: &PredictResponse, active: Option<usize>) -> Vec<DrawBox> {
        filter_detections(&p.boxes, self.play_conf, self.play_effective_class())
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
                let title = if self.play_compare.enabled {
                    "Copied A/B compare JSON"
                } else {
                    "Copied detection JSON"
                };
                self.toast(Tone::Success, title, Some(summary.into()), cx);
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
                "Open images",
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
                "Open images",
                Btn::Secondary,
                Some("icons/image-up.svg"),
                window,
                cx,
                act(|this, w, cx| this.pick_image(w, cx)),
            )
            .tooltip(crate::controls::tip("Open one or more images", Some("Ctrl O")))
            .into_any_element()
        };
        let folder = if let Some(why) = self.open_image_blocked() {
            self.button_disabled(
                "play-open-folder",
                "Folder",
                Btn::Secondary,
                Some("icons/folder-open.svg"),
                why,
                window,
                cx,
            )
            .into_any_element()
        } else {
            self.button(
                "play-open-folder",
                "Folder",
                Btn::Secondary,
                Some("icons/folder-open.svg"),
                window,
                cx,
                act(|this, w, cx| this.pick_play_folder(w, cx)),
            )
            .tooltip(crate::controls::tip("Open a folder of images as a batch", None))
            .into_any_element()
        };
        let header = page_header(
            &t,
            "Playground",
            if no_models_hdr {
                "Run a model on any image."
            } else if self.play_compare.enabled {
                "Compare two models side-by-side on any image. Drop files or a folder anywhere in the window."
            } else {
                "Run a model on any image or folder batch. Drop files or a folder anywhere in the window."
            },
            // No models: the body's call to action is the only one (no inert button beside it).
            if no_models_hdr { Vec::new() } else { vec![compare_toggle, open, folder] },
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
        let eff_class = self.play_effective_class();
        let n = self
            .play_pred
            .ready()
            .map_or(0, |p| filter_detections(&p.boxes, self.play_conf, eff_class).len());
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

#[cfg(test)]
mod tests {
    use crate::app::Route;
    use crate::app::tests::{drive, open};
    use crate::store::Load;

    #[test]
    fn play_compare_class_filter_keeps_keyboard_focus_during_pending_batch_reload() {
        use folo_ui::{KeyDownEvent, Keystroke, PlatformInput};

        let (mut app, window) = open();
        app.update(crate::app::bind_keys);
        drive(&mut app, window, |this, w, cx| {
            this.navigate(Route::Playground, w, cx);
            this.set_play_compare(true, cx);
        });
        for _ in 0..4 {
            app.frame(window.into());
            app.run_until_idle();
        }
        drive(&mut app, window, |this, _, cx| {
            let ds = this.play_sample_dataset().expect("sample dataset");
            this.play_dataset_image(&ds, "img_00801", cx);
        });
        for _ in 0..4 {
            app.frame(window.into());
            app.run_until_idle();
        }
        // Focus the "Only A" chip in Compare Delta, then simulate an in-flight batch/rerun reload.
        drive(&mut app, window, |this, w, cx| {
            this.focus_key("play-cmp-cls-a", w, cx);
            this.kb = true;
            this.play_pred = Load::Loading;
            this.play_pred_pending = true;
            this.play_compare.pred_b = Load::Loading;
            this.play_compare.pred_b_pending = true;
            cx.notify();
        });
        app.frame(window.into());
        // Pressing Tab while responses are delayed must stay inside Compare controls (Only B),
        // not drop focus back to Overview in the sidebar.
        app.simulate_input(
            window.into(),
            PlatformInput::KeyDown(KeyDownEvent {
                keystroke: Keystroke::parse("tab").unwrap(),
                is_held: false,
            }),
        );
        app.run_until_idle();
        app.frame(window.into());
        drive(&mut app, window, |this, w, cx| {
            assert!(
                this.fh("play-cmp-cls-b", cx).is_focused(w),
                "expected Tab to advance to Only B chip inside Compare rail"
            );
            assert!(
                !this.fh("nav-Overview", cx).is_focused(w),
                "focus must not jump back to sidebar Overview during pending A/B reload"
            );
        });
    }
}
