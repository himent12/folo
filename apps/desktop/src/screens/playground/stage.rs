//! The stage: the image with its boxes, zoom and pan, and dropping or opening an image.

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Bounds, Context, InteractiveElement, IntoElement, MouseMoveEvent, ParentElement,
    StatefulInteractiveElement, Styled, Window, canvas, div, px,
};

use crate::app::{FoloApp, IMAGE_FORMATS, act};
use crate::controls::Btn;
use crate::store::Load;
use crate::theme::{radius, theme, ty};
use crate::ui::*;
use crate::viz::{Fit, hit_box, paint_image_with_boxes, to_image};

impl FoloApp {
    pub(super) fn stage(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let no_models = self.store.models.ready().is_some_and(|m| m.is_empty());
        if self.play_image.is_none() {
            // Every enabled action does something: with no datasets either, the first step is one.
            let no_datasets = self.store.datasets.ready().is_some_and(|d| d.is_empty());
            let datasets_failed = self.store.datasets.error().is_some();
            let datasets_loading = self.store.datasets.ready().is_none() && !datasets_failed;
            // The model list failed: the step is fetching it again, not picking an image.
            let models_failed = self.store.models.error().is_some();
            // The side panel offers validation images only from a dataset that has val samples.
            let samples =
                self.play_sample_dataset().is_some_and(|ds_id| match self.store.images.get(&ds_id) {
                    Some(Load::Ready(v)) => !super::samples::pick_val_samples(v, 1).is_empty(),
                    Some(Load::Failed(_)) => false,
                    _ => true,
                });
            let pick_disabled = !models_failed && no_models && datasets_loading;
            if pick_disabled && self.fh("play-pick", cx).is_focused(window) {
                window.focus(&self.fh("play-pick-disabled", cx));
            } else if !pick_disabled && self.fh("play-pick-disabled", cx).is_focused(window) {
                window.focus(&self.fh("play-pick", cx));
            }
            let pick = if models_failed {
                self.button(
                    "play-pick",
                    "Retry",
                    Btn::Primary,
                    Some("icons/refresh-cw.svg"),
                    window,
                    cx,
                    act(|this, _, cx| this.retry_models(cx)),
                )
            } else if no_models && datasets_failed {
                self.button(
                    "play-pick",
                    "Retry datasets",
                    Btn::Primary,
                    Some("icons/refresh-cw.svg"),
                    window,
                    cx,
                    act(|this, _, cx| this.retry_datasets(cx)),
                )
            } else if no_models && datasets_loading {
                self.button_disabled(
                    "play-pick",
                    "New run",
                    Btn::Primary,
                    Some("icons/plus.svg"),
                    "Loading datasets…",
                    window,
                    cx,
                )
            } else if no_models && no_datasets {
                self.button(
                    "play-pick",
                    "Add dataset",
                    Btn::Primary,
                    Some("icons/plus.svg"),
                    window,
                    cx,
                    act(|this, w, cx| {
                        // The label promises the picker: land on Datasets with it open (the built-in demo
                        // can't register folders, and says so there instead).
                        this.navigate(crate::app::Route::Datasets, w, cx);
                        this.add_dataset_folder(cx);
                    }),
                )
            } else if no_models {
                self.button(
                    "play-pick",
                    "New run",
                    Btn::Primary,
                    Some("icons/plus.svg"),
                    window,
                    cx,
                    act(|this, w, cx| this.open_new_run(None, w, cx)),
                )
            } else {
                self.button(
                    "play-pick",
                    "Choose images",
                    Btn::Primary,
                    Some("icons/image-up.svg"),
                    window,
                    cx,
                    act(|this, w, cx| this.pick_image(w, cx)),
                )
            };
            let demo_batch_btn = (!no_models && !models_failed && samples).then(|| {
                let label =
                    if self.server_mode == crate::app::ServerMode::Mock { "Demo batch" } else { "Val batch" };
                self.button(
                    "play-empty-batch",
                    label,
                    Btn::Secondary,
                    Some("icons/images.svg"),
                    window,
                    cx,
                    act(|this, w, cx| this.load_play_demo_batch(w, cx)),
                )
                .tooltip(crate::controls::tip("Load sample validation images into the batch filmstrip", None))
            });
            let drop_hover = self.play_drop_hover;
            // Capped at 420 px and centred: an empty state, not a 970 px hole.
            let zone = div()
                .id("drop-zone")
                .flex_1()
                .min_w_0()
                .rounded(px(radius::LG))
                // The panel surface, like every other empty state.
                .bg(t.bg_panel)
                .border_1()
                // Dashed only where a drop works (a model to run it).
                .when(!no_models, |d| d.border_dashed())
                .border_color(if drop_hover { t.accent_line } else { t.line_strong })
                .flex()
                .items_center()
                .justify_center()
                .on_drag_move(cx.listener(|this, _: &folo_ui::DragMoveEvent<folo_ui::ExternalPaths>, _, cx| {
                    if !this.play_drop_hover {
                        this.play_drop_hover = true;
                        cx.notify();
                    }
                }))
                .child(
                    div()
                        .flex()
                        .flex_col()
                        .items_center()
                        .gap(px(16.))
                        .child(crate::app::logo(&t, 40.))
                        .child(
                            div()
                                .flex()
                                .flex_col()
                                .items_center()
                                .gap(px(4.))
                                // Headline, body and button name the same next step.
                                .child(text(ty::LG, t.text_primary).child(if models_failed {
                                    "The model list didn't load"
                                } else if no_models && datasets_failed {
                                    "The dataset list didn't load"
                                } else if no_models && no_datasets {
                                    "Add a dataset to get started"
                                } else if no_models {
                                    "Train a model to start detecting"
                                } else {
                                    "Drop an image to detect objects"
                                }))
                                .child(text(ty::SM, t.text_secondary).child(if models_failed || (no_models && datasets_failed) {
                                    "Check the connection, then try again.".to_string()
                                } else if no_models && no_datasets {
                                    "Then train a model on it and come back with an image.".to_string()
                                } else if no_models {
                                    "The Playground runs your trained models. Start a run, then come back with an image."
                                        .to_string()
                                } else if samples {
                                    format!("{IMAGE_FORMATS} · or pick a validation image on the right")
                                } else {
                                    IMAGE_FORMATS.to_string()
                                })),
                        )
                        .child(
                            div()
                                .flex()
                                .flex_wrap()
                                .justify_center()
                                .items_center()
                                .gap(px(10.))
                                .child(pick)
                                .children(demo_batch_btn)
                                .when(!no_models && !models_failed, |d| {
                                    d.child(
                                        div()
                                            .flex()
                                            .items_center()
                                            .gap(px(6.))
                                            .child(text(ty::SM, t.text_tertiary).child("or press"))
                                            .child(shortcut(&t, &["Ctrl", "O"])),
                                    )
                                }),
                        ),
                )
                .max_h(px(420.));
            let stage_w = {
                let rail = if window.viewport_size().width < px(1100.) { 280. } else { 320. };
                (self.content_width(window) - rail - 16. - 2.).max(0.)
            };
            let batch_strip =
                self.play_batch.is_active().then(|| self.play_batch_filmstrip(&t, stage_w, window, cx));
            let err_banner = self.play_batch_failed_banner(&t, window, cx);
            return div()
                .flex_1()
                .min_w_0()
                .flex()
                .flex_col()
                .gap(px(10.))
                .children(batch_strip)
                .children(err_banner)
                .child(zone)
                .into_any_element();
        }
        if self.play_batch_qa_zero_match() {
            let stage_w = {
                let rail = if window.viewport_size().width < px(1100.) { 280. } else { 320. };
                (self.content_width(window) - rail - 16. - 2.).max(0.)
            };
            let batch_strip = self.play_batch_filmstrip(&t, stage_w, window, cx);
            return div()
                .id("stage")
                .flex_1()
                .min_w_0()
                .flex()
                .flex_col()
                .rounded(px(radius::LG))
                .border_1()
                .border_color(t.line_default)
                .overflow_hidden()
                .child(
                    div()
                        .flex_1()
                        .min_h_0()
                        .bg(t.bg_panel)
                        .flex()
                        .items_center()
                        .justify_center()
                        .child(empty_state(
                            &t,
                            "icons/list-filter.svg",
                            "No batch images match this QA filter",
                            "Lower confidence, switch QA filter chips, or click Show all below to inspect images.",
                            Tone::Neutral,
                        )),
                )
                .child(batch_strip)
                .into_any_element();
        }
        if self.play_compare.enabled && !no_models {
            return self.compare_stage(window, cx);
        }
        let active = self.play_active(window, cx);
        let img = self.play_image.as_ref().unwrap();
        let dims = (img.width as f32, img.height as f32);
        let image = img.image.clone();
        let name = img.name.clone();
        // The toolbar sheds by priority, not by flex order: the whole "dataset / file" with the
        // image's size if both fit; else the whole name without the size; else the file alone.
        // Whatever is shed moves into the file's tip. Widths are shaped in the crumb's own font,
        // Geist Mono as file names are everywhere (fallbacks included, so CJK names measure true).
        // `room` is the stage less every other item, measured below.
        let size_text = format!("{} × {}", img.width, img.height);
        let (prefix_all, file) = match name.rsplit_once(" / ") {
            Some((p, f)) => (Some(p.to_string()), f.to_string()),
            None => (None, name.to_string()),
        };
        let file_w = f32::from(crate::charts::mono_width(window, &file, ty::SM.size));
        // The dataset in the UI sans (a name, as everywhere else), the file in mono; the crumb's
        // "/" sits between 6 px gaps, a little wider than the shaped " / ".
        let full_w = file_w
            + prefix_all.as_ref().map_or(0., |p| {
                f32::from(crate::charts::text_width(window, &format!("{p} / "), ty::SM.size))
            })
            + 6.;
        let stage_w = {
            let rail = if window.viewport_size().width < px(1100.) { 280. } else { 320. };
            // Stage width inside its 2 px border.
            (self.content_width(window) - rail - 16. - 2.).max(0.)
        };
        let bar_gap = if stage_w < 520. { 8. } else { 10. };
        let sim = self.play_target.is_none() && !self.store.api.local_inference();
        let latency_label = self.play_pred.ready().map(|p| latency_text(p.timing.total_ms, sim));
        let (boxes, count, loading) = match &self.play_pred {
            Load::Ready(p) => {
                let b = self.visible_boxes(p, active);
                let n = b.len();
                (b, Some(n), false)
            }
            Load::Loading => (Vec::new(), None, true),
            _ => (Vec::new(), None, false),
        };
        let count_label = count.map(|n| {
            format!("{n} {} ≥ {}", if n == 1 { "detection" } else { "detections" }, fmt_conf(self.play_conf))
        });
        let mut measure = |s: &str| f32::from(crate::charts::text_width(window, s, ty::SM.size));
        // Always-present chrome: px(12) padding (24), 4 gaps between [icon, file, spacer, rule,
        // zoom_bar], image icon (14), spacer min (8), divider rule (1), zoom_bar (help 24 +
        // zoom-out 28 + Fit 48 + zoom-in 28 + 3*2 gaps = 134), plus 8 px safety margin.
        let base_chrome = 24. + 4. * bar_gap + 14. + 8. + 1. + 134. + 8.;
        let loading_cost = if loading { measure("Detecting…") + bar_gap } else { 0. };
        // Shed secondary readouts before the filename or zoom controls ever clip: count first
        // (it already sits in the side rail), then latency (or loading on sub-minimum stages).
        let min_file = file_w.min(52.);
        let mut avail = (stage_w - base_chrome).max(0.);
        let show_loading = loading && avail - loading_cost >= min_file;
        if show_loading {
            avail -= loading_cost;
        }
        let lat_cost = latency_label.as_deref().map_or(0., |s| measure(s) + 1. + 2. * bar_gap);
        let show_latency = latency_label.is_some() && avail - lat_cost >= min_file;
        if show_latency {
            avail -= lat_cost;
        }
        let cnt_cost = count_label.as_deref().map_or(0., |s| measure(s) + 1. + 2. * bar_gap);
        let show_count = (show_latency || latency_label.is_none())
            && count_label.is_some()
            && avail - cnt_cost >= min_file;
        if show_count {
            avail -= cnt_cost;
        }
        let room = avail;
        const SIZE_W: f32 = 68.;
        let (show_prefix, show_size) = match prefix_all {
            None => (false, file_w + SIZE_W <= room),
            Some(_) if full_w + SIZE_W <= room => (true, true),
            Some(_) if full_w <= room => (true, false),
            Some(_) => (false, false),
        };
        let prefix = prefix_all.clone().filter(|_| show_prefix);
        let file_budget =
            (room - if show_size { SIZE_W } else { 0. } - if show_prefix { full_w - file_w } else { 0. })
                .clamp(0., 520.);
        // The tip carries everything the toolbar shed: the dataset, the size, a cut file name,
        // and any readout shed on narrow stages.
        let mut tip_parts =
            vec![if prefix_all.is_some() && !show_prefix { name.to_string() } else { file.clone() }];
        if !show_size {
            tip_parts.push(size_text.clone());
        }
        if loading && !show_loading {
            tip_parts.push("Detecting…".to_string());
        }
        if !show_latency && let Some(lat) = latency_label.as_ref() {
            tip_parts.push(lat.clone());
        }
        if !show_count && let Some(cnt) = count_label.as_ref() {
            tip_parts.push(cnt.clone());
        }
        let tip_text = tip_parts.join(" · ");
        let file_tip = file_w > file_budget
            || !show_size
            || (prefix_all.is_some() && !show_prefix)
            || (loading && !show_loading)
            || (latency_label.is_some() && !show_latency)
            || (count_label.is_some() && !show_count);
        let bounds_cell = self.play_view.clone();
        if let Some(b) = bounds_cell.get() {
            self.play_pan =
                crate::viz::clamp_pan(b, dims.0, dims.1, Fit::Contain, self.play_zoom, self.play_pan);
        }
        let paint_cell = bounds_cell.clone();
        let wheel_cell = bounds_cell.clone();
        let (zoom, pan) = (self.play_zoom, self.play_pan);
        // Fitted (by the readout, 0, −, or a new image), the readout is text: focus left on it
        // moves to its neighbour instead of falling back to the window.
        if zoom <= 1.0 && self.fh("play-fit", cx).is_focused(window) {
            window.focus(&self.fh("play-zoom-in", cx));
        }
        // How to move around the image, as the inspector lists it (the fitted "Fit" names no key).
        let view_help = self.help_icon(
            "play-view-help",
            "Scroll, + or − zoom · drag, or arrows when zoomed, pan · 0 fits the image",
            window,
            cx,
        );
        // The image is its own Tab stop, and a press on it focuses it: the arrows then pan the
        // image (zoomed) rather than stepping whichever slider had focus last.
        let canvas_fh = self.fh("stage-canvas", cx);
        let canvas_focused = canvas_fh.is_focused(window);
        let canvas_ring = self.ring(&canvas_fh, window);
        let zoom_bar = div()
            .flex()
            .items_center()
            .gap(px(2.))
            .child(div().pr(px(4.)).child(view_help))
            .child(self.icon_button(
                "play-zoom-out",
                "icons/zoom-out.svg",
                "Zoom out (−)",
                window,
                cx,
                act(|this, _, cx| this.play_zoom_by(1.0 / 1.4, cx)),
            ))
            .child(self.zoom_readout(
                "play-fit",
                zoom,
                window,
                cx,
                act(|this, _, cx| {
                    this.play_zoom = 1.0;
                    this.play_pan = (0., 0.);
                    this.play_drag = None;
                    cx.notify();
                }),
            ))
            .child(self.icon_button(
                "play-zoom-in",
                "icons/zoom-in.svg",
                "Zoom in (+)",
                window,
                cx,
                act(|this, _, cx| this.play_zoom_by(1.4, cx)),
            ));
        let draw_boxes = if self.play_boxes { boxes } else { Vec::new() };
        let hover_boxes = draw_boxes.clone();
        let paint_boxes = draw_boxes;
        let show_labels = self.play_labels;
        let batch_strip =
            self.play_batch.is_active().then(|| self.play_batch_filmstrip(&t, stage_w, window, cx));
        let tt = t;
        div()
            .id("stage")
            .flex_1()
            .min_w_0()
            .flex()
            .flex_col()
            .rounded(px(radius::LG))
            .border_1()
            .border_color(t.line_default)
            .overflow_hidden()
            .child(
                div()
                    .h(px(40.))
                    .flex_none()
                    .px(px(12.))
                    .flex()
                    .items_center()
                    .gap(px(bar_gap))
                    .bg(t.bg_panel)
                    .border_b_1()
                    .border_color(t.line_subtle)
                    .child(div().flex_none().child(icon("icons/image.svg", 14., t.text_tertiary)))
                    // "dataset / file", by the priority above; the file name is never shed.
                    .map(|d| {
                        d.child(
                            div()
                                .flex()
                                .items_center()
                                .gap(px(6.))
                                .min_w_0()
                                .flex_shrink()
                                // The dataset is a name (UI sans, as everywhere); the file is a path
                                // (Geist Mono).
                                .when_some(prefix, |d, p| {
                                    d.child(
                                        text(ty::SM, t.text_tertiary)
                                            .min_w_0()
                                            .flex_shrink()
                                            .ellipsis()
                                            .child(p),
                                    )
                                    .child(text(ty::SM, t.text_tertiary).flex_none().child("/"))
                                })
                                .child(
                                    // A long file name ends in "…" at its known budget, as truncation needs.
                                    // What the toolbar shed shows on hover and on keyboard focus.
                                    self.disclose(
                                        div().id("play-file-name"),
                                        "play-file-name",
                                        tip_text.clone(),
                                        file_tip,
                                        28.,
                                        true,
                                        window,
                                        cx,
                                    )
                                    .flex_none()
                                    .child(
                                        text(ty::SM, t.text_primary)
                                            .font_family(crate::theme::MONO)
                                            .w(px(file_w.min(file_budget).ceil()))
                                            .ellipsis()
                                            .child(file.clone()),
                                    ),
                                ),
                        )
                    })
                    .when(show_size, |d| {
                        d.child(text(ty::SM, t.text_tertiary).flex_none().tnum().child(size_text.clone()))
                    })
                    .child(div().flex_1().min_w(px(8.)))
                    .when(show_loading, |d| {
                        d.child(text(ty::SM, t.text_tertiary).flex_none().child("Detecting…"))
                    })
                    // The latency always reads here when it fits, beside the count (the rail's
                    // breakdown is below the fold at most window sizes).
                    // The built-in demo's number is simulated, and says so wherever it shows.
                    .when_some(latency_label.filter(|_| show_latency), |d, lat| {
                        // One rule between every group, the image's own (name, size) included, so
                        // a narrow toolbar's spacer never reads as a missing divider.
                        d.child(div().flex_none().w(px(1.)).h(px(16.)).bg(t.line_subtle))
                            .child(text(ty::SM, t.text_secondary).flex_none().tnum().child(lat))
                    })
                    .when_some(count_label.filter(|_| show_count), |d, cnt| {
                        d.child(div().flex_none().w(px(1.)).h(px(16.)).bg(t.line_subtle)).child(
                            div()
                                .flex_none()
                                .flex()
                                .items_center()
                                .gap(px(6.))
                                .child(text(ty::SM, t.text_secondary).tnum().child(cnt)),
                        )
                    })
                    .child(div().flex_none().w(px(1.)).h(px(16.)).bg(t.line_subtle))
                    .child(zoom_bar.flex_none()),
            )
            .child(
                div()
                    .id("stage-canvas")
                    .track_focus(&canvas_fh)
                    .relative()
                    .flex_1()
                    .min_h_0()
                    .bg(t.bg_inset)
                    .cursor(if zoom > 1.0 {
                        folo_ui::CursorStyle::OpenHand
                    } else {
                        folo_ui::CursorStyle::Arrow
                    })
                    .on_scroll_wheel(cx.listener(move |this, ev: &folo_ui::ScrollWheelEvent, w, cx| {
                        let Some(b) = wheel_cell.get() else { return };
                        let dy = f32::from(ev.delta.pixel_delta(w.line_height()).y);
                        let new_zoom =
                            (this.play_zoom * (1.0 + dy * 0.0025).clamp(0.5, 2.0)).clamp(1.0, 12.0);
                        this.play_pan = crate::viz::zoom_about(
                            b,
                            dims,
                            this.play_zoom,
                            this.play_pan,
                            ev.position,
                            new_zoom,
                        );
                        this.play_zoom = new_zoom;
                        this.play_drag = None;
                        w.prevent_default();
                        cx.stop_propagation();
                        cx.notify();
                    }))
                    .on_mouse_down(
                        folo_ui::MouseButton::Left,
                        cx.listener(move |this, ev: &folo_ui::MouseDownEvent, w, _| {
                            w.focus(&canvas_fh);
                            this.play_drag = Some((ev.position, this.play_pan));
                        }),
                    )
                    .on_mouse_up(
                        folo_ui::MouseButton::Left,
                        cx.listener(|this, _, _, _| this.play_drag = None),
                    )
                    .on_mouse_up_out(
                        folo_ui::MouseButton::Left,
                        cx.listener(|this, _, _, _| this.play_drag = None),
                    )
                    .on_mouse_move(cx.listener(move |this, ev: &MouseMoveEvent, _, cx| {
                        if let Some(drag) = this.play_drag {
                            if ev.pressed_button == Some(folo_ui::MouseButton::Left) {
                                if this.play_zoom > 1.0 {
                                    let (next_pan, next_drag) = clamp_and_rebase_drag(
                                        drag,
                                        ev.position,
                                        bounds_cell.get(),
                                        dims,
                                        this.play_zoom,
                                    );
                                    this.play_pan = next_pan;
                                    this.play_drag = Some(next_drag);
                                    cx.notify();
                                    return;
                                }
                            } else {
                                this.play_drag = None;
                            }
                        }
                        let Some(b) = bounds_cell.get() else { return };
                        let hit = if b.contains(&ev.position) {
                            let (x, y) =
                                to_image(b, dims, Fit::Contain, this.play_zoom, this.play_pan, ev.position);
                            hit_box(&hover_boxes, x, y)
                        } else {
                            None
                        };
                        if hit != this.play_hover || (hit.is_some() && !this.play_hover_from_stage) {
                            this.play_hover = hit;
                            this.play_hover_from_stage = hit.is_some();
                            // The pointer moved over the image: hover leads again.
                            this.kb = false;
                            cx.notify();
                        }
                    }))
                    .on_hover(cx.listener(|this, h: &bool, _, cx| {
                        if !*h && this.play_hover_from_stage {
                            this.play_hover_from_stage = false;
                            if this.play_hover.take().is_some() {
                                cx.notify();
                            }
                        }
                    }))
                    .child(
                        div()
                            .size_full()
                            .when(self.play_pred_dim && self.play_pred_pending, |d| d.opacity(0.5))
                            .child(
                                canvas(
                                    |_, _, _| (),
                                    move |b, _, window, cx| {
                                        let pad_x = (b.size.width / 2.).min(px(16.));
                                        let pad_y = (b.size.height / 2.).min(px(16.));
                                        let inner = Bounds::new(
                                            b.origin + folo_ui::point(pad_x, pad_y),
                                            b.size - folo_ui::size(pad_x * 2., pad_y * 2.),
                                        );
                                        paint_cell.set(Some(inner));
                                        paint_image_with_boxes(
                                            &tt,
                                            window,
                                            cx,
                                            inner,
                                            Some(&image),
                                            dims,
                                            &paint_boxes,
                                            Fit::Contain,
                                            zoom,
                                            pan,
                                            show_labels,
                                            false,
                                            0.,
                                        );
                                    },
                                )
                                .size_full(),
                            ),
                    )
                    // Focused, the stage says so: a focus-coloured inset edge (2 px from the
                    // keyboard, like every ring; 1 px after a click), and from the keyboard, its keys.
                    .when(canvas_focused, |d| {
                        d.child(
                            div()
                                .absolute()
                                .inset_0()
                                .when(canvas_ring, |d| d.border_2())
                                .when(!canvas_ring, |d| d.border_1())
                                .border_color(t.focus),
                        )
                    })
                    .when(canvas_ring, |d| {
                        self.hint_keys(if zoom > 1.0 {
                            if self.play_batch.is_active() {
                                "Arrows pan · Shift ×4 · [ ] batch · + − zoom · 0 fits"
                            } else {
                                "Arrows pan · Shift ×4 · + − zoom · 0 fits"
                            }
                        } else if self.play_batch.is_active() {
                            "← → or [ ] batch · + − zoom · arrows pan when zoomed · 0 fits"
                        } else {
                            "+ − zoom · arrows pan when zoomed · 0 fits"
                        });
                        d
                    }),
            )
            .children(self.play_batch_failed_banner(&t, window, cx))
            .children(batch_strip)
            .into_any_element()
    }
}

/// Active Playground pointer drag anchor: `(origin_pointer, origin_pan)`.
pub(super) type PlayDrag = (folo_ui::Point<folo_ui::Pixels>, (f32, f32));

/// Applies a held-drag pointer move to Playground pan, clamps to the canvas bounds, and rebases
/// any clamped axis so reversing direction past a pan limit moves immediately without a dead zone.
pub(super) fn clamp_and_rebase_drag(
    (from, pan0): PlayDrag,
    pos: folo_ui::Point<folo_ui::Pixels>,
    bounds: Option<Bounds<folo_ui::Pixels>>,
    dims: (f32, f32),
    zoom: f32,
) -> ((f32, f32), PlayDrag) {
    let raw = (pan0.0 + f32::from(pos.x - from.x), pan0.1 + f32::from(pos.y - from.y));
    let next = match bounds {
        Some(b) => crate::viz::clamp_pan(b, dims.0, dims.1, Fit::Contain, zoom, raw),
        None => raw,
    };
    let clamped_x = (next.0 - raw.0).abs() > 1e-4;
    let clamped_y = (next.1 - raw.1).abs() > 1e-4;
    let next_from =
        folo_ui::point(if clamped_x { pos.x } else { from.x }, if clamped_y { pos.y } else { from.y });
    let next_pan0 = (if clamped_x { next.0 } else { pan0.0 }, if clamped_y { next.1 } else { pan0.1 });
    (next, (next_from, next_pan0))
}
