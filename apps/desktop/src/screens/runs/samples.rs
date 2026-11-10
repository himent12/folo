//! A run's validation samples panel.

use std::rc::Rc;

use gpui::{
    AnyElement, Context, InteractiveElement, IntoElement, ParentElement, SharedString,
    StatefulInteractiveElement, Styled, Window, div, px,
};

use crate::api::Run;
use crate::app::{FoloApp, SampleView, act};
use crate::screens::image_canvas;
use crate::store::{ImgKind, Load};
use crate::theme::{radius, theme, ty};
use crate::ui::*;
use crate::viz::{DrawBox, Fit};

impl FoloApp {
    pub(super) fn samples_panel(
        &mut self,
        r: &Run,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = theme(cx);
        // The samples come from each epoch's evaluation: training images when there's no val split.
        let images = if self.eval_split(r) == "train" { "training" } else { "validation" };
        let views = [SampleView::Predictions, SampleView::Truth, SampleView::Both];
        let sel = views.iter().position(|v| *v == self.sample_view).unwrap_or(0);
        let seg = self.segmented(
            "samples-view",
            &["Predictions", "Ground truth", "Both"],
            sel,
            window,
            cx,
            Rc::new(move |this, i, _, _| this.sample_view = views[i]),
        );
        let classes =
            self.store.dataset_summary(&r.config.dataset_id).map(|d| d.classes.clone()).unwrap_or_default();
        let body = match self.store.samples.get(&r.id) {
            _ if r.epoch == 0 => div()
                .child(empty_state(
                    &t,
                    "icons/image.svg",
                    "No evaluation yet",
                    &format!(
                        "Predictions on 8 of the 16 fixed {images} images appear after the first epoch."
                    ),
                    Tone::Neutral,
                ))
                .into_any_element(),
            Some(Load::Ready(s)) => {
                let samples: Vec<crate::api::Sample> = s.samples.iter().take(8).cloned().collect();
                let epoch = s.epoch;
                let view = self.sample_view;
                let tiles: Vec<AnyElement> = samples
                    .iter()
                    .enumerate()
                    .map(|(n, smp)| {
                        let img = self.image(&smp.dataset_id, &smp.image_id, ImgKind::Thumb, cx);
                        let mut boxes: Vec<DrawBox> = Vec::new();
                        if view != SampleView::Predictions {
                            boxes.extend(smp.truth.iter().map(|b| DrawBox {
                                truth: true,
                                ..DrawBox::new(b.class, b.x1, b.y1, b.x2, b.y2)
                            }));
                        }
                        if view != SampleView::Truth {
                            boxes.extend(smp.pred.iter().filter(|p| p.score >= 0.3).map(|p| DrawBox {
                                // Class and score where the cell has room (viz drops the score below 140 px).
                                label: classes.get(p.class as usize).cloned().map(Into::into),
                                score: Some(p.score),
                                ..DrawBox::new(p.class, p.x1, p.y1, p.x2, p.y2)
                            }));
                        }
                        let dims = (smp.width as f32, smp.height as f32);
                        // Square cells: every sample gets the same frame whatever its aspect. The
                        // ratio sits on an inner full-width box, whose width is definite.
                        let mut square = div()
                            .w_full()
                            .rounded(px(radius::MD))
                            .overflow_hidden()
                            .bg(t.bg_inset)
                            .child(image_canvas(&t, img, dims, boxes, Fit::Cover, true, false));
                        square.style().aspect_ratio = Some(1.0);
                        // Each sample is a Tab stop: click or Enter opens it in the inspector (the
                        // image in full, its labels, and the dataset to walk from it).
                        let (ds, image, run) = (smp.dataset_id.clone(), smp.image_id.clone(), r.id.clone());
                        if self.ring(&self.fh(format!("smp-{n}"), cx), window) {
                            self.hint_keys(
                                "Enter opens the sample in the inspector, with this epoch's predictions",
                            );
                        }
                        let tile = div()
                            .id(SharedString::from(format!("smp-{n}")))
                            .flex_1()
                            .min_w(px(0.))
                            .rounded(px(radius::MD))
                            .cursor_pointer()
                            .tooltip(crate::controls::tip("Open in the inspector", None))
                            .child(
                                square
                                    .relative()
                                    // Over an image a fill wouldn't show: the edge answers instead
                                    // (1 px on hover, 2 px while pressed, inside the tile so nothing
                                    // moves).
                                    .child(
                                        div()
                                            .id(SharedString::from(format!("smp-edge-{n}")))
                                            .absolute()
                                            .inset_0()
                                            .rounded(px(radius::MD))
                                            .border_2()
                                            .border_color(gpui::transparent_black())
                                            .group_hover(SharedString::from(format!("smp-{n}")), |s| {
                                                s.border_1().border_color(t.line_strong)
                                            })
                                            .group_active(SharedString::from(format!("smp-{n}")), |s| {
                                                s.border_2().border_color(t.text_secondary)
                                            }),
                                    ),
                            )
                            .group(SharedString::from(format!("smp-{n}")));
                        self.bind(
                            tile,
                            &format!("smp-{n}"),
                            radius::MD,
                            window,
                            cx,
                            act(move |this, w, cx| {
                                this.open_sample(ds.clone(), image.clone(), run.clone(), w, cx)
                            }),
                        )
                        .into_any_element()
                    })
                    .collect();
                // One row of eight across the page's width (the log takes its own full-width row
                // below), about the size the 3:5 column's two rows of four had.
                div()
                    .flex()
                    .flex_col()
                    .gap(px(8.))
                    .p(px(16.))
                    .child(div().flex().items_start().gap(px(8.)).children(tiles))
                    // The caption says what the tiles show, view by view.
                    .child(text(ty::SM, t.text_tertiary).tnum().child(format!(
                        "Epoch {epoch} · 8 of 16 fixed {images} images · {}",
                        match view {
                            SampleView::Predictions => "predictions with score ≥ 0.30",
                            SampleView::Truth => "ground-truth labels",
                            SampleView::Both => "ground truth and predictions with score ≥ 0.30",
                        }
                    )))
                    .into_any_element()
            }
            Some(Load::Failed(e)) => div()
                .p(px(16.))
                .child(text(ty::SM, t.danger_text).child(e.message.clone()))
                .into_any_element(),
            _ => div()
                .p(px(12.))
                .flex()
                .gap(px(8.))
                .children((0..8).map(|i| {
                    let mut sq = div().w_full().child(skeleton(&t, format!("smp{i}")));
                    sq.style().aspect_ratio = Some(1.0);
                    div().flex_1().min_w(px(0.)).child(sq)
                }))
                .into_any_element(),
        };
        panel(&t)
            .flex_1()
            .flex()
            .flex_col()
            .child(
                panel_header(
                    &t,
                    if images == "training" { "Training samples" } else { "Validation samples" },
                )
                .child(seg),
            )
            .child(
                div().px(px(16.)).pt(px(10.)).flex().flex_wrap().gap(px(12.)).children(
                    self.store
                        .dataset_summary(&r.config.dataset_id)
                        .map(|d| d.classes.clone())
                        .unwrap_or_default()
                        .into_iter()
                        .enumerate()
                        .map(|(i, name)| {
                            div()
                                .flex()
                                .items_center()
                                .gap(px(6.))
                                .child(crate::ui::class_swatch(&t, i as u32, 8.))
                                .child(text(ty::SM, t.text_secondary).child(name))
                        }),
                ),
            )
            .child(body)
            .into_any_element()
    }
}
